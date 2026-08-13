use hashbrown::hash_map::RawEntryMut;
use rustc_hash::FxHasher;
use std::{
    hash::{BuildHasherDefault, Hash, Hasher},
    sync::Mutex,
};

use crate::{GreenNode, GreenNodeData, GreenToken, GreenTokenData, NodeOrToken, SyntaxKind};

use super::element::GreenElement;

type HashMap<K, V> = hashbrown::HashMap<K, V, BuildHasherDefault<FxHasher>>;
type NodeShard = Mutex<HashMap<NoHash<GreenNode>, ()>>;
type TokenShard = Mutex<HashMap<NoHash<GreenToken>, ()>>;

#[derive(Debug)]
struct NoHash<T>(T);

/// Interner for GreenTokens and GreenNodes.
#[derive(Default, Debug)]
pub struct NodeCache {
    nodes: HashMap<NoHash<GreenNode>, ()>,
    tokens: HashMap<NoHash<GreenToken>, ()>,
}

const SHARD_COUNT: usize = 256;
// Clear at hashbrown's 7/8 load limit for 16K buckets instead of resizing to 32K.
const NODE_CAPACITY_PER_SHARD: usize = 14 * 1024;
const TOKEN_CAPACITY_PER_SHARD: usize = 4 * 1024;
const MAX_SHARED_NODE_CHILDREN: usize = 1;
const MAX_SHARED_TOKEN_LEN: usize = 8;

/// A bounded, thread-safe interner for sharing immutable green trees across builders.
///
/// Individual shards are cleared at their capacity so unused trees cannot accumulate without
/// bound. Hash matches are confirmed with structural equality before a tree is reused.
#[derive(Debug)]
pub struct SharedNodeCache {
    nodes: Box<[NodeShard]>,
    tokens: Box<[TokenShard]>,
}

impl Default for SharedNodeCache {
    fn default() -> Self {
        let nodes = (0..SHARD_COUNT).map(|_| Mutex::new(HashMap::default())).collect();
        let tokens = (0..SHARD_COUNT).map(|_| Mutex::new(HashMap::default())).collect();
        SharedNodeCache { nodes, tokens }
    }
}

fn token_hash(token: &GreenTokenData) -> u64 {
    let mut h = FxHasher::default();
    token.kind().hash(&mut h);
    token.text().hash(&mut h);
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

fn element_id(element: NodeOrToken<&GreenNodeData, &GreenTokenData>) -> *const () {
    match element {
        NodeOrToken::Node(it) => it as *const GreenNodeData as *const (),
        NodeOrToken::Token(it) => it as *const GreenTokenData as *const (),
    }
}

impl SharedNodeCache {
    /// Drops all cached nodes and tokens.
    ///
    /// Green trees already returned by builders remain valid.
    pub fn clear(&self) {
        for shard in &self.nodes {
            *shard.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = HashMap::default();
        }
        for shard in &self.tokens {
            *shard.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = HashMap::default();
        }
    }

    fn node(
        &self,
        hash: u64,
        kind: SyntaxKind,
        children: &[(u64, u64, GreenElement)],
    ) -> Option<GreenNode> {
        let shard = self.nodes[hash as usize % SHARD_COUNT]
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        shard
            .raw_entry()
            .from_hash(hash, |cached| node_matches(&cached.0, kind, children))
            .map(|(cached, ())| cached.0.clone())
    }

    fn insert_node(&self, hash: u64, node: GreenNode) -> GreenNode {
        let mut shard = self.nodes[hash as usize % SHARD_COUNT]
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if shard.capacity() == 0 {
            *shard = HashMap::with_capacity_and_hasher(
                NODE_CAPACITY_PER_SHARD,
                BuildHasherDefault::default(),
            );
        }
        if shard.len() >= NODE_CAPACITY_PER_SHARD {
            if let Some((cached, ())) = shard.raw_entry().from_hash(hash, |cached| cached.0 == node)
            {
                return cached.0.clone();
            }
            shard.clear();
        }
        match shard.raw_entry_mut().from_hash(hash, |cached| cached.0 == node) {
            RawEntryMut::Occupied(entry) => entry.key().0.clone(),
            RawEntryMut::Vacant(entry) => {
                entry.insert_with_hasher(hash, NoHash(node.clone()), (), |cached| {
                    node_hash(&cached.0)
                });
                node
            }
        }
    }

    fn token(&self, hash: u64, kind: SyntaxKind, text: &str) -> Option<GreenToken> {
        let shard = self.tokens[hash as usize % SHARD_COUNT]
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        shard
            .raw_entry()
            .from_hash(hash, |cached| cached.0.kind() == kind && cached.0.text() == text)
            .map(|(cached, ())| cached.0.clone())
    }

    fn insert_token(&self, hash: u64, token: GreenToken) -> GreenToken {
        let mut shard = self.tokens[hash as usize % SHARD_COUNT]
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if shard.capacity() == 0 {
            *shard = HashMap::with_capacity_and_hasher(
                TOKEN_CAPACITY_PER_SHARD,
                BuildHasherDefault::default(),
            );
        }
        if shard.len() >= TOKEN_CAPACITY_PER_SHARD {
            if let Some((cached, ())) =
                shard.raw_entry().from_hash(hash, |cached| cached.0 == token)
            {
                return cached.0.clone();
            }
            shard.clear();
        }
        match shard.raw_entry_mut().from_hash(hash, |cached| cached.0 == token) {
            RawEntryMut::Occupied(entry) => entry.key().0.clone(),
            RawEntryMut::Vacant(entry) => {
                entry.insert_with_hasher(hash, NoHash(token.clone()), (), |cached| {
                    token_hash(&cached.0)
                });
                token
            }
        }
    }
}

fn node_matches(
    node: &GreenNodeData,
    kind: SyntaxKind,
    children: &[(u64, u64, GreenElement)],
) -> bool {
    node.kind() == kind
        && node.children().len() == children.len()
        && node.children().zip(children).all(|(left, (_, _, right))| {
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
        children: &mut Vec<(u64, u64, GreenElement)>,
        first_child: usize,
        shared_cache: Option<&SharedNodeCache>,
    ) -> (u64, u64, GreenNode) {
        let build_node = move |children: &mut Vec<(u64, u64, GreenElement)>| {
            GreenNode::new(kind, children.drain(first_child..).map(|(_, _, it)| it))
        };

        let children_ref = &children[first_child..];
        let structural_hash = {
            let mut h = FxHasher::default();
            kind.hash(&mut h);
            for &(_, hash, _) in children_ref {
                hash.hash(&mut h);
            }
            h.finish()
        };

        if let Some(cache) = shared_cache {
            if children_ref.len() > MAX_SHARED_NODE_CHILDREN {
                return (0, structural_hash, build_node(children));
            }
            if let Some(node) = cache.node(structural_hash, kind, children_ref) {
                drop(children.drain(first_child..));
                return (0, structural_hash, node);
            }
            let node = cache.insert_node(structural_hash, build_node(children));
            return (0, structural_hash, node);
        }

        if children_ref.len() > 3 {
            return (0, structural_hash, build_node(children));
        }

        let hash = {
            let mut h = FxHasher::default();
            kind.hash(&mut h);
            for &(hash, _, _) in children_ref {
                if hash == 0 {
                    return (0, structural_hash, build_node(children));
                }
                hash.hash(&mut h);
            }
            h.finish()
        };

        let entry = self.nodes.raw_entry_mut().from_hash(hash, |node| {
            node.0.kind() == kind && node.0.children().len() == children_ref.len() && {
                let lhs = node.0.children().map(element_id);
                let rhs = children_ref.iter().map(|(_, _, it)| element_id(it.as_deref()));
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
                entry.insert_with_hasher(hash, NoHash(node.clone()), (), |cached| {
                    node_hash(&cached.0)
                });
                node
            }
        };

        (hash, structural_hash, node)
    }

    pub(crate) fn token(
        &mut self,
        kind: SyntaxKind,
        text: &str,
        shared_cache: Option<&SharedNodeCache>,
    ) -> (u64, GreenToken) {
        let hash = {
            let mut h = FxHasher::default();
            kind.hash(&mut h);
            text.hash(&mut h);
            h.finish()
        };

        if let Some(cache) = shared_cache {
            if text.len() > MAX_SHARED_TOKEN_LEN {
                return (hash, GreenToken::new(kind, text));
            }
            let token = cache
                .token(hash, kind, text)
                .unwrap_or_else(|| cache.insert_token(hash, GreenToken::new(kind, text)));
            return (hash, token);
        }

        let entry = self
            .tokens
            .raw_entry_mut()
            .from_hash(hash, |token| token.0.kind() == kind && token.0.text() == text);

        let token = match entry {
            RawEntryMut::Occupied(entry) => entry.key().0.clone(),
            RawEntryMut::Vacant(entry) => {
                let token = GreenToken::new(kind, text);
                entry.insert_with_hasher(hash, NoHash(token.clone()), (), |cached| {
                    token_hash(&cached.0)
                });
                token
            }
        };

        (hash, token)
    }

    pub(crate) fn token_from_green(&mut self, token: GreenToken) -> (u64, GreenToken) {
        (token_hash(&token), token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_cache_checks_equality_after_hash_match() {
        let cache = SharedNodeCache::default();
        let first = cache.insert_node(0, GreenNode::new(SyntaxKind(1), []));
        let second = cache.insert_node(0, GreenNode::new(SyntaxKind(2), []));
        let repeated = cache.insert_node(0, GreenNode::new(SyntaxKind(2), []));

        assert!(!std::ptr::eq::<GreenNodeData>(&*first, &*second));
        assert!(std::ptr::eq::<GreenNodeData>(&*second, &*repeated));
    }

    #[test]
    fn shared_cache_reserves_bounded_shards_on_first_use() {
        let cache = SharedNodeCache::default();
        cache.insert_node(0, GreenNode::new(SyntaxKind(1), []));
        cache.insert_token(0, GreenToken::new(SyntaxKind(1), "x"));

        assert!(cache.nodes[0].lock().unwrap().capacity() >= NODE_CAPACITY_PER_SHARD);
        assert!(cache.tokens[0].lock().unwrap().capacity() >= TOKEN_CAPACITY_PER_SHARD);
    }
}
