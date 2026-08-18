use hashbrown::hash_map::RawEntryMut;
use rustc_hash::FxHasher;
use std::{
    hash::{BuildHasherDefault, Hash, Hasher},
    mem,
    sync::Mutex,
};

use crate::{
    green::{GreenChild, GreenElementRef},
    GreenNode, GreenNodeData, GreenToken, GreenTokenData, NodeOrToken, SyntaxKind,
};

use super::element::GreenElement;

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
enum SharedElement {
    Local,
    Shareable { retained_weight: u32 },
}

#[derive(Debug)]
pub(super) struct SharedCache<'cache> {
    local: NodeCache,
    shared: &'cache SharedNodeCache,
    elements: Vec<SharedElement>,
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
// Keep shard bits above the low bits used to address our largest
// hashbrown table, and below hashbrown's high-bit h2 fingerprint.
const SHARD_SHIFT: u32 = 16;
const SHARD_BITS: u32 = SHARD_COUNT.trailing_zeros();
const _: () = {
    assert!(SHARD_COUNT.is_power_of_two());
    assert!(NODE_CAPACITY_PER_SHARD < 1usize << SHARD_SHIFT);
    assert!(TOKEN_CAPACITY_PER_SHARD < 1usize << SHARD_SHIFT);
    assert!(SHARD_SHIFT + SHARD_BITS <= usize::BITS - 7);
};

/// An opt-in, bounded interner for sharing immutable green descendants across builders.
///
/// Use this cache with [`crate::SharedGreenNodeBuilder`]. Ordinary builders and
/// [`NodeCache`] retain their local-only behavior. Finished trees have distinct root allocations,
/// while eligible descendants may be shared.
///
/// Individual shards grow on demand and rotate when their aggregate accounted retained weight
/// reaches its budget, with entry counts as a secondary guard. Across all shards, the accounted
/// budgets are at most 256 MiB for nodes and 16 MiB for tokens before eviction. These are retention
/// accounting limits, not exact allocator-byte guarantees. Parent weights conservatively include
/// retained descendants, and hash matches are confirmed with structural equality.
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
    token_hash_parts(token.kind(), token.text())
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
            NodeOrToken::Node(node) => node_hash(node),
            NodeOrToken::Token(token) => token_hash(token),
        }
        .hash(&mut h);
    }
    h.finish()
}

fn cached_node_hash(kind: SyntaxKind, children: &[(u64, GreenElement)]) -> Option<u64> {
    let mut h = FxHasher::default();
    kind.hash(&mut h);
    for child in children {
        if child.0 == 0 {
            return None;
        }
        child.0.hash(&mut h);
    }
    Some(h.finish())
}

fn node_weight(child_count: usize, elements: &[SharedElement]) -> Option<u32> {
    let node = mem::size_of::<GreenNode>()
        .saturating_add(child_count.saturating_mul(mem::size_of::<GreenChild>()))
        .min(u32::MAX as usize) as u32;
    elements.iter().try_fold(node, |weight, element| match element {
        SharedElement::Local => None,
        SharedElement::Shareable { retained_weight } => {
            Some(weight.saturating_add(*retained_weight))
        }
    })
}

fn token_weight(text: &str) -> u32 {
    token_weight_for_len(text.len())
}

fn token_weight_for_len(text_len: usize) -> u32 {
    mem::size_of::<GreenToken>().saturating_add(text_len).min(u32::MAX as usize) as u32
}

fn element_id(element: GreenElementRef<'_>) -> *const () {
    match element {
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
        children: &mut Vec<(u64, GreenElement)>,
        first_child: usize,
        retained_weight: u32,
    ) -> GreenNode {
        let mut shard =
            self.nodes[shard_index(hash)].lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some((cached, ())) = shard.entries.raw_entry().from_hash(hash, |cached| {
            cached.hash == hash && node_matches(&cached.value, kind, &children[first_child..])
        }) {
            drop(children.drain(first_child..));
            return cached.value.clone();
        }
        let old = (shard.entries.len() >= NODE_CAPACITY_PER_SHARD
            || shard.retained_weight.saturating_add(retained_weight)
                > NODE_RETAINED_WEIGHT_PER_SHARD)
            .then(|| mem::take(&mut *shard));
        shard.retained_weight = shard.retained_weight.saturating_add(retained_weight);
        let node = GreenNode::new(kind, children.drain(first_child..).map(|child| child.1));
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

fn node_matches(node: &GreenNodeData, kind: SyntaxKind, children: &[(u64, GreenElement)]) -> bool {
    node.kind() == kind
        && node.children().len() == children.len()
        && node.children().zip(children).all(|(left, right)| {
            let right = &right.1;
            if element_id(left) == element_id(right.as_deref()) {
                return true;
            }
            match (left, right.as_deref()) {
                (NodeOrToken::Node(left), NodeOrToken::Node(right)) => left == right,
                (NodeOrToken::Token(left), NodeOrToken::Token(right)) => left == right,
                _ => false,
            }
        })
}

impl NodeCache {
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
}

impl<'cache> SharedCache<'cache> {
    pub(super) fn new(shared: &'cache SharedNodeCache) -> Self {
        SharedCache { local: NodeCache::default(), shared, elements: Vec::new() }
    }

    pub(super) fn finish(self, root: GreenNode) -> GreenNode {
        match self.elements.as_slice() {
            [SharedElement::Local] => root,
            [SharedElement::Shareable { .. }] => {
                GreenNode::new(root.kind(), root.children().map(|child| child.to_owned()))
            }
            _ => unreachable!(),
        }
    }

    #[inline]
    pub(super) fn node(
        &mut self,
        kind: SyntaxKind,
        children: &mut Vec<(u64, GreenElement)>,
        first_child: usize,
    ) -> (u64, GreenNode) {
        let children_ref = &children[first_child..];
        if children_ref.len() <= MAX_SHARED_NODE_CHILDREN {
            if let Some(retained_weight) =
                node_weight(children_ref.len(), &self.elements[first_child..])
            {
                if retained_weight <= MAX_SHARED_SUBTREE_WEIGHT {
                    if let Some(hash) = cached_node_hash(kind, children_ref) {
                        let node =
                            self.shared.node(hash, kind, children, first_child, retained_weight);
                        self.elements.truncate(first_child);
                        self.elements.push(SharedElement::Shareable { retained_weight });
                        return (hash, node);
                    }
                }
            }
        }

        let node = self.local.node(kind, children, first_child);
        self.elements.truncate(first_child);
        self.elements.push(SharedElement::Local);
        node
    }

    #[inline]
    pub(super) fn token(&mut self, kind: SyntaxKind, text: &str) -> (u64, GreenToken) {
        let retained_weight = token_weight(text);
        if text.len() <= MAX_SHARED_TOKEN_LEN {
            let hash = token_hash_parts(kind, text);
            let token = self.shared.token(hash, kind, text, retained_weight);
            self.elements.push(SharedElement::Shareable { retained_weight });
            return (hash, token);
        }

        let token = self.local.token(kind, text);
        self.elements.push(SharedElement::Local);
        token
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SharedGreenNodeBuilder;
    use std::{sync::Arc, thread};

    fn build(cache: &SharedNodeCache) -> GreenNode {
        let mut builder = SharedGreenNodeBuilder::new(cache);
        builder.start_node(SyntaxKind(0));
        builder.token(SyntaxKind(1), "one");
        builder.finish_node();
        builder.finish()
    }

    #[test]
    fn local_cache_returns_hashes_separately_from_values() {
        let mut cache = NodeCache::default();
        let (hash, token): (u64, GreenToken) = cache.token(SyntaxKind(1), "one");
        let mut children = vec![(hash, token.into())];
        let (_, _node): (u64, GreenNode) = cache.node(SyntaxKind(0), &mut children, 0);

        assert_eq!(mem::size_of::<NoHash<GreenNode>>(), mem::size_of::<GreenNode>());
        assert_eq!(mem::size_of::<NoHash<GreenToken>>(), mem::size_of::<GreenToken>());
    }

    #[test]
    fn token_weight_saturates() {
        assert_eq!(token_weight_for_len(usize::MAX), u32::MAX);
    }

    #[test]
    fn node_hash_requires_cached_child_hashes() {
        let child = (0, GreenToken::new(SyntaxKind(1), "one").into());

        assert_eq!(cached_node_hash(SyntaxKind(0), &[child]), None);
    }

    #[test]
    fn shared_fallback_does_not_mark_uncached_nodes_canonical() {
        let cache = SharedNodeCache::default();
        let mut shared = SharedCache::new(&cache);
        let mut children = Vec::new();
        for kind in 0..4 {
            let (hash, token) = shared.token(SyntaxKind(kind), "x");
            children.push((hash, token.into()));
        }

        let (wide_hash, wide) = shared.node(SyntaxKind(4), &mut children, 0);
        assert_eq!(wide_hash, 0);
        children.push((wide_hash, wide.into()));

        let (parent_hash, _) = shared.node(SyntaxKind(5), &mut children, 0);
        assert_eq!(parent_hash, 0);
    }

    #[test]
    fn shared_finish_returns_local_fallback_root_directly() {
        let cache = SharedNodeCache::default();
        let root = GreenNode::new(SyntaxKind(0), std::iter::empty());
        let original = root.clone();
        let shared = SharedCache {
            local: NodeCache::default(),
            shared: &cache,
            elements: vec![SharedElement::Local],
        };

        let finished = shared.finish(root);

        assert!(std::ptr::eq::<GreenNodeData>(&*original, &*finished));
    }

    #[test]
    fn shared_cache_checks_equality_after_hash_match() {
        let cache = SharedNodeCache::default();
        let mut children = Vec::new();
        let first = cache.node(0, SyntaxKind(1), &mut children, 0, 1);
        let second = cache.node(0, SyntaxKind(2), &mut children, 0, 1);
        let repeated = cache.node(0, SyntaxKind(2), &mut children, 0, 1);

        assert!(!std::ptr::eq::<GreenNodeData>(&*first, &*second));
        assert!(std::ptr::eq::<GreenNodeData>(&*second, &*repeated));
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
    fn shard_index_uses_middle_hash_bits() {
        assert_ne!(shard_index(0), shard_index(1 << 16));
    }

    #[test]
    fn shard_index_ignores_hashbrown_fingerprint_bits() {
        let middle = 0x5a << 16;

        assert_eq!(shard_index(middle), shard_index(middle | (0xfe << 56)));
    }

    #[test]
    fn shard_index_ignores_low_bucket_bits() {
        assert_eq!(shard_index(0), shard_index(0xffff));
    }

    #[test]
    fn shared_token_cache_checks_equality_after_hash_match() {
        let cache = SharedNodeCache::default();
        let first = cache.token(0, SyntaxKind(1), "one", 1);
        let second = cache.token(0, SyntaxKind(1), "two", 1);
        let repeated = cache.token(0, SyntaxKind(1), "two", 1);

        assert!(!std::ptr::eq(&*first, &*second));
        assert!(std::ptr::eq(&*second, &*repeated));
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
        let clearer = {
            let cache = Arc::clone(&cache);
            thread::spawn(move || {
                for _ in 0..100 {
                    cache.clear();
                }
            })
        };
        let builders: Vec<_> = (0..4)
            .map(|_| {
                let cache = Arc::clone(&cache);
                thread::spawn(move || {
                    for _ in 0..100 {
                        assert_eq!(build(&cache).to_string(), "one");
                    }
                })
            })
            .collect();

        clearer.join().unwrap();
        for builder in builders {
            builder.join().unwrap();
        }
    }
}
