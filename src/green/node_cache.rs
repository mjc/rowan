use hashbrown::hash_map::RawEntryMut;
use rustc_hash::FxHasher;
use std::{
    hash::{BuildHasherDefault, Hash, Hasher},
    mem,
    sync::Mutex,
};

use crate::{
    green::GreenElementRef, GreenNode, GreenNodeData, GreenToken, GreenTokenData, NodeOrToken,
    SyntaxKind,
};

use super::element::GreenElement;
use super::{node, token};

type HashMap<K, V> = hashbrown::HashMap<K, V, BuildHasherDefault<FxHasher>>;
type SharedNodeMap = HashMap<Hashed<GreenNode>, ()>;
type SharedTokenMap = HashMap<Hashed<GreenToken>, ()>;

#[derive(Debug, Default)]
struct NodeShardData {
    entries: SharedNodeMap,
    retained_weight: u32,
}

#[derive(Debug, Default)]
struct TokenShardData {
    entries: SharedTokenMap,
    retained_weight: u32,
}

type NodeShard = Mutex<NodeShardData>;
type TokenShard = Mutex<TokenShardData>;

#[derive(Debug)]
struct NoHash<T>(T);

#[derive(Debug)]
struct Hashed<T> {
    hash: u64,
    value: T,
}

#[derive(Debug)]
pub(super) struct CachedElement {
    hash: u64,
    interned: bool,
    shareable: bool,
    retained_weight: u32,
    pub(super) green: GreenElement,
}

#[derive(Debug)]
pub(super) struct SharedCacheBackend<'cache> {
    local: NodeCache,
    shared: &'cache SharedNodeCache,
    children: Vec<CachedElement>,
}

/// Interner for GreenTokens and GreenNodes
// XXX: the impl is a bit tricky. As usual when writing interners, we want to
// store all values in one HashSet.
//
// However, hashing trees is fun: hash of the tree is recursively defined. We
// maintain an invariant -- if the tree is interned, then all of its children
// are interned as well.
//
// That means that computing the hash naively is wasteful -- we just *know*
// hashes of children, and we can re-use those.
//
// So here we use *raw* API of hashbrown and provide the hashes manually,
// instead of going via a `Hash` impl. Our manual `Hash` and the
// `#[derive(Hash)]` are actually different! At some point we had a fun bug,
// where we accidentally mixed the two hashes, which made the cache much less
// efficient.
//
// To fix that, we additionally wrap the data in `NoHash` wrapper, to make sure
// we don't accidentally use the wrong hash!
#[derive(Default, Debug)]
pub struct NodeCache {
    nodes: HashMap<NoHash<GreenNode>, ()>,
    tokens: HashMap<NoHash<GreenToken>, ()>,
}

const SHARD_COUNT: usize = 256;
const NODE_CAPACITY_PER_SHARD: usize = 14 * 1024;
const TOKEN_CAPACITY_PER_SHARD: usize = 4 * 1024;
const NODE_RETAINED_WEIGHT_PER_SHARD: u32 = 1024 * 1024;
const TOKEN_RETAINED_WEIGHT_PER_SHARD: u32 = 64 * 1024;
const MAX_SHARED_NODE_CHILDREN: usize = 1;
const MAX_SHARED_TOKEN_LEN: usize = 8;
const MAX_SHARED_SUBTREE_WEIGHT: u32 = 1024;
const SHARD_BITS: u32 = SHARD_COUNT.trailing_zeros();
// Select from the high end of the hash while leaving hashbrown's top seven h2 bits intact.
const SHARD_SHIFT: u32 = u64::BITS - 7 - SHARD_BITS;
const _: () = {
    assert!(SHARD_COUNT.is_power_of_two());
    assert!(SHARD_SHIFT + SHARD_BITS <= u64::BITS - 7);
};

/// An opt-in, bounded interner for sharing immutable green descendants across builders.
///
/// Use this cache with [`crate::GreenNodeBuilder::with_shared_cache`]. Ordinary builders and
/// [`NodeCache`] retain their local-only behavior. Finished trees have distinct root allocations,
/// while eligible descendants may be shared.
///
/// Individual shards grow on demand and rotate when their aggregate accounted retained weight
/// reaches its budget, with entry counts as a secondary guard. Across all shards, the accounted
/// budgets are at most 256 MiB for nodes and 16 MiB for tokens before eviction. These are retention
/// accounting limits, not exact allocator-byte guarantees. Each weight includes the retained green
/// allocation and its hash-table entry storage; parent weights conservatively include retained
/// descendants. Hash matches are confirmed with structural equality.
#[derive(Debug)]
pub struct SharedNodeCache {
    nodes: Box<[NodeShard]>,
    tokens: Box<[TokenShard]>,
}

impl Default for SharedNodeCache {
    fn default() -> Self {
        let nodes = (0..SHARD_COUNT).map(|_| Mutex::new(NodeShardData::default())).collect();
        let tokens = (0..SHARD_COUNT).map(|_| Mutex::new(TokenShardData::default())).collect();
        SharedNodeCache { nodes, tokens }
    }
}

fn token_hash(token: &GreenTokenData) -> u64 {
    let mut h = FxHasher::default();
    token.kind().hash(&mut h);
    token.text().hash(&mut h);
    h.finish()
}

fn token_hash_parts(kind: SyntaxKind, text: &str) -> u64 {
    let mut h = FxHasher::default();
    kind.hash(&mut h);
    text.hash(&mut h);
    h.finish()
}

fn node_hash(node: &GreenNodeData) -> u64 {
    let mut h = FxHasher::default();
    node.kind().hash(&mut h);
    for child in node.children() {
        match child {
            NodeOrToken::Node(it) => node_hash(it),
            NodeOrToken::Token(it) => token_hash(it),
        }
        .hash(&mut h)
    }
    h.finish()
}

fn cached_node_hash(kind: SyntaxKind, children: &[CachedElement]) -> u64 {
    let mut h = FxHasher::default();
    kind.hash(&mut h);
    for child in children {
        child.hash.hash(&mut h);
    }
    h.finish()
}

fn node_weight(children: &[CachedElement]) -> Option<u32> {
    let child_count = children.len();
    let node = node::allocation_size(child_count)
        .saturating_add(cache_entry_size::<GreenNode>())
        .min(u32::MAX as usize) as u32;
    children.iter().try_fold(node, |weight, element| {
        element.shareable.then(|| weight.saturating_add(element.retained_weight))
    })
}

fn token_weight(text: &str) -> u32 {
    token_weight_for_len(text.len())
}

fn token_weight_for_len(text_len: usize) -> u32 {
    token::allocation_size(text_len)
        .saturating_add(cache_entry_size::<GreenToken>())
        .min(u32::MAX as usize) as u32
}

fn cache_entry_size<T>() -> usize {
    mem::size_of::<(Hashed<T>, ())>().saturating_add(1)
}

fn element_id(elem: GreenElementRef<'_>) -> *const () {
    match elem {
        NodeOrToken::Node(it) => it as *const GreenNodeData as *const (),
        NodeOrToken::Token(it) => it as *const GreenTokenData as *const (),
    }
}

#[inline]
fn shard_index(hash: u64) -> usize {
    ((hash >> SHARD_SHIFT) as usize) & (SHARD_COUNT - 1)
}

impl SharedNodeCache {
    /// Drops cache ownership present when the operation reaches each shard.
    ///
    /// Green trees already returned by builders remain valid. This is not a global atomic clear:
    /// builders running concurrently may repopulate shards before this method returns. Callers that
    /// require an empty cache must call it while builders using the cache are quiescent.
    pub fn clear(&self) {
        for shard in &self.nodes {
            let old = {
                let mut shard = shard.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                mem::take(&mut *shard)
            };
            drop(old);
        }
        for shard in &self.tokens {
            let old = {
                let mut shard = shard.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                mem::take(&mut *shard)
            };
            drop(old);
        }
    }

    fn node(
        &self,
        hash: u64,
        kind: SyntaxKind,
        children: &mut Vec<CachedElement>,
        first_child: usize,
        retained_weight: u32,
    ) -> GreenNode {
        let mut shard =
            self.nodes[shard_index(hash)].lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(cached) = shard
            .entries
            .raw_entry()
            .from_hash(hash, |cached| {
                cached.hash == hash && node_matches(&cached.value, kind, &children[first_child..])
            })
            .map(|(cached, ())| cached.value.clone())
        {
            drop(shard);
            drop(children.drain(first_child..));
            return cached;
        }
        let old = (shard.entries.len() >= NODE_CAPACITY_PER_SHARD
            || shard.retained_weight.saturating_add(retained_weight)
                > NODE_RETAINED_WEIGHT_PER_SHARD)
            .then(|| mem::take(&mut *shard));
        shard.retained_weight = shard.retained_weight.saturating_add(retained_weight);
        let node = GreenNode::new(kind, children.drain(first_child..).map(|child| child.green));
        let entry = match shard.entries.raw_entry_mut().from_hash(hash, |_| false) {
            RawEntryMut::Vacant(entry) => entry,
            RawEntryMut::Occupied(_) => unreachable!(),
        };
        entry.insert_with_hasher(hash, Hashed { hash, value: node.clone() }, (), |cached| {
            cached.hash
        });
        drop(shard);
        drop(old);
        node
    }
    fn token(&self, hash: u64, kind: SyntaxKind, text: &str, retained_weight: u32) -> GreenToken {
        let mut shard =
            self.tokens[shard_index(hash)].lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some((cached, ())) = shard.entries.raw_entry().from_hash(hash, |cached| {
            cached.hash == hash && cached.value.kind() == kind && cached.value.text() == text
        }) {
            return cached.value.clone();
        }
        let old = (shard.entries.len() >= TOKEN_CAPACITY_PER_SHARD
            || shard.retained_weight.saturating_add(retained_weight)
                > TOKEN_RETAINED_WEIGHT_PER_SHARD)
            .then(|| mem::take(&mut *shard));
        shard.retained_weight = shard.retained_weight.saturating_add(retained_weight);
        let token = GreenToken::new(kind, text);
        let entry = match shard.entries.raw_entry_mut().from_hash(hash, |_| false) {
            RawEntryMut::Vacant(entry) => entry,
            RawEntryMut::Occupied(_) => unreachable!(),
        };
        entry.insert_with_hasher(hash, Hashed { hash, value: token.clone() }, (), |cached| {
            cached.hash
        });
        drop(shard);
        drop(old);
        token
    }
}

fn node_matches(node: &GreenNodeData, kind: SyntaxKind, children: &[CachedElement]) -> bool {
    node.kind() == kind
        && node.children().len() == children.len()
        && node.children().zip(children).all(|(left, right)| {
            if element_id(left) == element_id(right.green.as_deref()) {
                return true;
            }
            match (left, right.green.as_deref()) {
                (NodeOrToken::Node(left), NodeOrToken::Node(right)) => left == right,
                (NodeOrToken::Token(left), NodeOrToken::Token(right)) => left == right,
                _ => false,
            }
        })
}

impl NodeCache {
    #[inline]
    pub(crate) fn node(
        &mut self,
        kind: SyntaxKind,
        children: &mut Vec<(u64, GreenElement)>,
        first_child: usize,
    ) -> (u64, GreenNode) {
        let build_node = move |children: &mut Vec<(u64, GreenElement)>| {
            GreenNode::new(kind, children.drain(first_child..).map(|(_, it)| it))
        };

        let children_ref = &children[first_child..];
        if children_ref.len() > 3 {
            let node = build_node(children);
            return (0, node);
        }

        let hash = {
            let mut h = FxHasher::default();
            kind.hash(&mut h);
            for &(hash, _) in children_ref {
                if hash == 0 {
                    let node = build_node(children);
                    return (0, node);
                }
                hash.hash(&mut h);
            }
            h.finish()
        };

        // Green nodes are fully immutable, so it's ok to deduplicate them.
        // This is the same optimization that Roslyn does
        // https://github.com/KirillOsenkov/Bliki/wiki/Roslyn-Immutable-Trees
        //
        // For example, all `#[inline]` in this file share the same green node!
        // For `libsyntax/parse/parser.rs`, measurements show that deduping saves
        // 17% of the memory for green nodes!

        let entry = self.nodes.raw_entry_mut().from_hash(hash, |node| {
            node.0.kind() == kind && node.0.children().len() == children_ref.len() && {
                let lhs = node.0.children();
                let rhs = children_ref.iter().map(|(_, it)| it.as_deref());

                let lhs = lhs.map(element_id);
                let rhs = rhs.map(element_id);

                lhs.eq(rhs)
            }
        });

        let node = match entry {
            RawEntryMut::Occupied(entry) => {
                drop(children.drain(first_child..));
                entry.key().0.clone()
            }
            RawEntryMut::Vacant(entry) => {
                let node = build_node(children);
                entry.insert_with_hasher(hash, NoHash(node.clone()), (), |n| node_hash(&n.0));
                node
            }
        };

        (hash, node)
    }

    #[inline]
    pub(crate) fn token(&mut self, kind: SyntaxKind, text: &str) -> (u64, GreenToken) {
        let hash = {
            let mut h = FxHasher::default();
            kind.hash(&mut h);
            text.hash(&mut h);
            h.finish()
        };

        let entry = self
            .tokens
            .raw_entry_mut()
            .from_hash(hash, |token| token.0.kind() == kind && token.0.text() == text);
        let token = match entry {
            RawEntryMut::Occupied(entry) => entry.key().0.clone(),
            RawEntryMut::Vacant(entry) => {
                let token = GreenToken::new(kind, text);
                entry.insert_with_hasher(hash, NoHash(token.clone()), (), |t| token_hash(&t.0));
                token
            }
        };

        (hash, token)
    }

    fn shared_node(
        &mut self,
        kind: SyntaxKind,
        children: &mut Vec<CachedElement>,
        first_child: usize,
    ) -> CachedElement {
        let build_node = move |children: &mut Vec<CachedElement>| {
            GreenNode::new(kind, children.drain(first_child..).map(|child| child.green))
        };

        let children_ref = &children[first_child..];
        if children_ref.len() > 3 {
            let node = build_node(children);
            return CachedElement::local(0, false, node.into());
        }

        let mut h = FxHasher::default();
        kind.hash(&mut h);
        for child in children_ref {
            if !child.interned {
                let node = build_node(children);
                return CachedElement::local(0, false, node.into());
            }
            child.hash.hash(&mut h);
        }
        let hash = h.finish();

        // Green nodes are fully immutable, so it's ok to deduplicate them.
        // This is the same optimization that Roslyn does
        // https://github.com/KirillOsenkov/Bliki/wiki/Roslyn-Immutable-Trees
        //
        // For example, all `#[inline]` in this file share the same green node!
        // For `libsyntax/parse/parser.rs`, measurements show that deduping saves
        // 17% of the memory for green nodes!
        let entry = self.nodes.raw_entry_mut().from_hash(hash, |node| {
            node.0.kind() == kind && node.0.children().len() == children_ref.len() && {
                let lhs = node.0.children();
                let rhs = children_ref.iter().map(|child| child.green.as_deref());

                let lhs = lhs.map(element_id);
                let rhs = rhs.map(element_id);

                lhs.eq(rhs)
            }
        });

        let node = match entry {
            RawEntryMut::Occupied(entry) => {
                drop(children.drain(first_child..));
                entry.key().0.clone()
            }
            RawEntryMut::Vacant(entry) => {
                let node = build_node(children);
                entry.insert_with_hasher(hash, NoHash(node.clone()), (), |n| node_hash(&n.0));
                node
            }
        };

        CachedElement::local(hash, true, node.into())
    }

    fn shared_token(&mut self, kind: SyntaxKind, text: &str) -> CachedElement {
        let hash = {
            let mut h = FxHasher::default();
            kind.hash(&mut h);
            text.hash(&mut h);
            h.finish()
        };

        let entry = self
            .tokens
            .raw_entry_mut()
            .from_hash(hash, |token| token.0.kind() == kind && token.0.text() == text);

        let token = match entry {
            RawEntryMut::Occupied(entry) => entry.key().0.clone(),
            RawEntryMut::Vacant(entry) => {
                let token = GreenToken::new(kind, text);
                entry.insert_with_hasher(hash, NoHash(token.clone()), (), |t| token_hash(&t.0));
                token
            }
        };

        CachedElement::local(hash, true, token.into())
    }
}

impl CachedElement {
    fn local(hash: u64, interned: bool, green: GreenElement) -> Self {
        CachedElement { hash, interned, shareable: false, retained_weight: 0, green }
    }
}

impl<'cache> SharedCacheBackend<'cache> {
    pub(super) fn new(cache: &'cache SharedNodeCache) -> Self {
        SharedCacheBackend { local: NodeCache::default(), shared: cache, children: Vec::new() }
    }

    pub(super) fn len(&self) -> usize {
        self.children.len()
    }

    // Keep shared-only policy out of the ordinary builder's inlined hot path.
    #[inline(never)]
    pub(super) fn node(&mut self, kind: SyntaxKind, first_child: usize) {
        let children_ref = &self.children[first_child..];
        if children_ref.len() <= MAX_SHARED_NODE_CHILDREN {
            if let Some(retained_weight) = node_weight(children_ref) {
                if retained_weight <= MAX_SHARED_SUBTREE_WEIGHT {
                    let hash = cached_node_hash(kind, children_ref);
                    let node = self.shared.node(
                        hash,
                        kind,
                        &mut self.children,
                        first_child,
                        retained_weight,
                    );
                    self.children.push(CachedElement {
                        hash,
                        interned: true,
                        shareable: true,
                        retained_weight,
                        green: node.into(),
                    });
                    return;
                }
            }
        }

        let node = self.local.shared_node(kind, &mut self.children, first_child);
        self.children.push(node);
    }

    #[inline(never)]
    pub(super) fn token(&mut self, kind: SyntaxKind, text: &str) {
        if text.len() <= MAX_SHARED_TOKEN_LEN {
            let hash = token_hash_parts(kind, text);
            let retained_weight = token_weight(text);
            let token = self.shared.token(hash, kind, text, retained_weight);
            self.children.push(CachedElement {
                hash,
                interned: true,
                shareable: true,
                retained_weight,
                green: token.into(),
            });
        } else {
            self.children.push(self.local.shared_token(kind, text));
        }
    }

    pub(super) fn finish(mut self) -> GreenNode {
        assert_eq!(self.children.len(), 1);
        let root = self.children.pop().unwrap();
        let shareable = root.shareable;
        let node = match root.green {
            NodeOrToken::Node(node) => node,
            NodeOrToken::Token(_) => panic!(),
        };
        if shareable {
            GreenNode::new(node.kind(), node.children().map(|child| child.to_owned()))
        } else {
            node
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GreenNodeBuilder;
    use std::{
        sync::{
            atomic::{AtomicBool, AtomicUsize, Ordering},
            Arc, Barrier,
        },
        thread,
    };

    fn build(cache: &SharedNodeCache) -> GreenNode {
        let mut builder = GreenNodeBuilder::with_shared_cache(cache);
        builder.start_node(SyntaxKind(0));
        builder.token(SyntaxKind(1), "one");
        builder.finish_node();
        builder.finish()
    }

    #[test]
    fn local_cache_tracks_hashes_without_changing_map_entry_layout() {
        let mut cache = NodeCache::default();
        let token = cache.shared_token(SyntaxKind(1), "one");
        let mut children = vec![token];
        let node = cache.shared_node(SyntaxKind(0), &mut children, 0);

        assert!(node.interned);
        assert_eq!(mem::size_of::<NoHash<GreenNode>>(), mem::size_of::<GreenNode>());
        assert_eq!(mem::size_of::<NoHash<GreenToken>>(), mem::size_of::<GreenToken>());
        #[cfg(target_pointer_width = "64")]
        assert_eq!(mem::size_of::<CachedElement>(), 32);
    }

    #[test]
    fn token_weight_saturates() {
        assert_eq!(token_weight_for_len(usize::MAX), u32::MAX);
    }

    #[test]
    fn retained_weights_include_allocation_and_cache_entry_storage() {
        assert!(token_weight_for_len(0) > mem::size_of::<GreenToken>() as u32);
        assert!(node_weight(&[]).unwrap() > mem::size_of::<GreenNode>() as u32);
    }

    #[test]
    fn shared_backend_records_production_retained_weights() {
        let cache = SharedNodeCache::default();
        let mut backend = SharedCacheBackend::new(&cache);
        backend.token(SyntaxKind(1), "token");

        let token = &backend.children[0];
        assert_eq!(token.retained_weight, token_weight("token"));
        assert_eq!(
            cache.tokens[shard_index(token.hash)].lock().unwrap().retained_weight,
            token.retained_weight
        );

        let node_hash = cached_node_hash(SyntaxKind(2), &backend.children);
        let node_weight = node_weight(&backend.children).unwrap();
        backend.node(SyntaxKind(2), 0);

        assert_eq!(backend.children[0].retained_weight, node_weight);
        assert_eq!(
            cache.nodes[shard_index(node_hash)].lock().unwrap().retained_weight,
            node_weight
        );
    }

    #[test]
    fn shared_backend_rotates_token_shards_by_production_weight() {
        let cache = SharedNodeCache::default();
        let mut backend = SharedCacheBackend::new(&cache);
        let mut expected_weight: u32 = 0;

        for index in 0_u32.. {
            let text = format!("{index:08x}");
            let hash = token_hash_parts(SyntaxKind(1), &text);
            if shard_index(hash) != 0 {
                continue;
            }

            let weight = token_weight(&text);
            let rotated = expected_weight.saturating_add(weight) > TOKEN_RETAINED_WEIGHT_PER_SHARD;
            if rotated {
                expected_weight = 0;
            }
            expected_weight += weight;

            backend.token(SyntaxKind(1), &text);
            let token = backend.children.pop().unwrap();
            assert_eq!(token.retained_weight, weight);
            assert_eq!(cache.tokens[0].lock().unwrap().retained_weight, expected_weight);

            if rotated {
                break;
            }
        }
    }

    #[test]
    fn shared_fallback_does_not_mark_uncached_nodes_canonical() {
        let cache = SharedNodeCache::default();
        let mut shared = SharedCacheBackend::new(&cache);
        for kind in 0..4 {
            shared.token(SyntaxKind(kind), "x");
        }

        shared.node(SyntaxKind(4), 0);
        assert!(!shared.children[0].shareable);

        shared.node(SyntaxKind(5), 0);
        assert!(!shared.children[0].shareable);
    }

    #[test]
    fn shared_finish_returns_local_fallback_root_directly() {
        let cache = SharedNodeCache::default();
        let root = GreenNode::new(SyntaxKind(0), std::iter::empty());
        let original = root.clone();
        let mut shared = SharedCacheBackend::new(&cache);
        shared.children.push(CachedElement::local(0, false, root.into()));

        let finished = shared.finish();

        assert!(std::ptr::eq::<GreenNodeData>(&*original, &*finished));
    }

    #[test]
    fn shared_cache_checks_equality_after_hash_match() {
        let cache = SharedNodeCache::default();
        let mut children = Vec::new();
        let first = cache.node(0, SyntaxKind(1), &mut children, 0, 1);
        let second = cache.node(0, SyntaxKind(2), &mut children, 0, 1);
        let repeated = cache.node(0, SyntaxKind(2), &mut children, 0, 1);
        let first_repeated = cache.node(0, SyntaxKind(1), &mut children, 0, 1);

        assert!(!std::ptr::eq::<GreenNodeData>(&*first, &*second));
        assert!(std::ptr::eq::<GreenNodeData>(&*second, &*repeated));
        assert!(std::ptr::eq::<GreenNodeData>(&*first, &*first_repeated));
    }

    #[test]
    fn shared_cache_grows_shards_on_demand() {
        let cache = SharedNodeCache::default();
        cache.node(0, SyntaxKind(1), &mut Vec::new(), 0, 1);
        cache.token(0, SyntaxKind(1), "x", 1);

        let node_capacity = cache.nodes[0].lock().unwrap().entries.capacity();
        let token_capacity = cache.tokens[0].lock().unwrap().entries.capacity();
        assert!((1..NODE_CAPACITY_PER_SHARD).contains(&node_capacity));
        assert!((1..TOKEN_CAPACITY_PER_SHARD).contains(&token_capacity));
    }

    #[test]
    fn shared_cache_rotates_node_shards_by_retained_weight() {
        let cache = SharedNodeCache::default();
        let first =
            cache.node(0, SyntaxKind(1), &mut Vec::new(), 0, NODE_RETAINED_WEIGHT_PER_SHARD);
        let second = cache.node(0, SyntaxKind(2), &mut Vec::new(), 0, 1);

        assert!(!std::ptr::eq::<GreenNodeData>(&*first, &*second));
        let shard = cache.nodes[0].lock().unwrap();
        assert_eq!(shard.entries.len(), 1);
        assert_eq!(shard.retained_weight, 1);
    }

    #[test]
    fn shared_cache_rotates_token_shards_by_retained_weight() {
        let cache = SharedNodeCache::default();
        let first = cache.token(0, SyntaxKind(1), "one", TOKEN_RETAINED_WEIGHT_PER_SHARD);
        let second = cache.token(0, SyntaxKind(2), "two", 1);

        assert!(!std::ptr::eq(&*first, &*second));
        let shard = cache.tokens[0].lock().unwrap();
        assert_eq!(shard.entries.len(), 1);
        assert_eq!(shard.retained_weight, 1);
    }

    #[test]
    fn shared_cache_rotates_node_shards_by_entry_count() {
        let cache = SharedNodeCache::default();
        for kind in 0..=NODE_CAPACITY_PER_SHARD {
            cache.node(kind as u64, SyntaxKind(kind as u16), &mut Vec::new(), 0, 1);
        }

        let shard = cache.nodes[0].lock().unwrap();
        assert_eq!(shard.entries.len(), 1);
        assert_eq!(shard.retained_weight, 1);
    }

    #[test]
    fn shared_cache_rotates_token_shards_by_entry_count() {
        let cache = SharedNodeCache::default();
        for index in 0..=TOKEN_CAPACITY_PER_SHARD {
            cache.token(index as u64, SyntaxKind(1), &index.to_string(), 1);
        }

        let shard = cache.tokens[0].lock().unwrap();
        assert_eq!(shard.entries.len(), 1);
        assert_eq!(shard.retained_weight, 1);
    }

    #[test]
    fn shared_cache_is_empty_after_quiescent_clear() {
        let cache = SharedNodeCache::default();
        cache.node(0, SyntaxKind(1), &mut Vec::new(), 0, 1);
        cache.token(0, SyntaxKind(1), "one", 1);

        cache.clear();

        assert!(cache.nodes.iter().all(|shard| {
            let shard = shard.lock().unwrap();
            shard.entries.is_empty() && shard.retained_weight == 0
        }));
        assert!(cache.tokens.iter().all(|shard| {
            let shard = shard.lock().unwrap();
            shard.entries.is_empty() && shard.retained_weight == 0
        }));
    }

    #[test]
    fn shard_index_uses_high_hash_bits_below_the_fingerprint() {
        assert_ne!(shard_index(0), shard_index(1 << 49));
    }

    #[test]
    fn shard_index_ignores_hashbrown_fingerprint_bits() {
        let middle = 0x5a << 49;

        assert_eq!(shard_index(middle), shard_index(middle | (0xfe << 56)));
    }

    #[test]
    fn shard_index_ignores_low_bucket_bits() {
        assert_eq!(shard_index(0), shard_index((1 << 49) - 1));
    }

    #[test]
    fn shared_token_cache_checks_equality_after_hash_match() {
        let cache = SharedNodeCache::default();
        let first = cache.token(0, SyntaxKind(1), "one", 1);
        let second = cache.token(0, SyntaxKind(1), "two", 1);
        let repeated = cache.token(0, SyntaxKind(1), "two", 1);
        let first_repeated = cache.token(0, SyntaxKind(1), "one", 1);

        assert!(!std::ptr::eq(&*first, &*second));
        assert!(std::ptr::eq(&*second, &*repeated));
        assert!(std::ptr::eq(&*first, &*first_repeated));
    }

    #[test]
    fn shared_cache_reuses_trees_concurrently() {
        let cache = Arc::new(SharedNodeCache::default());
        let trees: Vec<_> = (0..8)
            .map(|_| {
                let cache = Arc::clone(&cache);
                thread::spawn(move || build(&cache))
            })
            .map(|thread| thread.join().unwrap())
            .collect();

        assert!(trees.iter().all(|tree| tree == &trees[0]));
        assert!(trees
            .iter()
            .skip(1)
            .all(|tree| !std::ptr::eq::<GreenNodeData>(&**tree, &*trees[0])));
        let tokens: Vec<_> = trees
            .iter()
            .map(|tree| tree.children().next().unwrap().into_token().unwrap())
            .collect();
        assert!(tokens.iter().all(|token| std::ptr::eq(*token, tokens[0])));
    }

    #[test]
    fn shared_cache_clear_is_safe_while_building() {
        let cache = Arc::new(SharedNodeCache::default());
        let start = Arc::new(Barrier::new(5));
        let active_builders = Arc::new(AtomicUsize::new(0));
        let completed_builds = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let clearer = {
            let cache = Arc::clone(&cache);
            let start = Arc::clone(&start);
            let active_builders = Arc::clone(&active_builders);
            let completed_builds = Arc::clone(&completed_builds);
            let stop = Arc::clone(&stop);
            thread::spawn(move || {
                start.wait();
                while completed_builds.load(Ordering::Acquire) < 4 {
                    thread::yield_now();
                }
                for _ in 0..100 {
                    cache.clear();
                    assert_eq!(active_builders.load(Ordering::Acquire), 4);
                }
                stop.store(true, Ordering::Release);
            })
        };
        let builders: Vec<_> = (0..4)
            .map(|_| {
                let cache = Arc::clone(&cache);
                let start = Arc::clone(&start);
                let active_builders = Arc::clone(&active_builders);
                let completed_builds = Arc::clone(&completed_builds);
                let stop = Arc::clone(&stop);
                thread::spawn(move || {
                    active_builders.fetch_add(1, Ordering::Release);
                    start.wait();
                    while !stop.load(Ordering::Acquire) {
                        assert_eq!(build(&cache).to_string(), "one");
                        completed_builds.fetch_add(1, Ordering::Release);
                    }
                    active_builders.fetch_sub(1, Ordering::Release);
                })
            })
            .collect();

        clearer.join().unwrap();
        for builder in builders {
            builder.join().unwrap();
        }
    }
}
