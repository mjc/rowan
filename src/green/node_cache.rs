use hashbrown::hash_map::RawEntryMut;
use rustc_hash::FxHasher;
use std::{
    fmt,
    hash::{BuildHasherDefault, Hash, Hasher},
    marker::PhantomData,
    mem::ManuallyDrop,
    ptr::{self, NonNull},
    sync::Mutex,
};

use crate::{GreenNode, GreenNodeData, GreenToken, GreenTokenData, NodeOrToken, SyntaxKind};

use super::element::GreenElement;

type HashMap<K, V> = hashbrown::HashMap<K, V, BuildHasherDefault<FxHasher>>;
type SharedSlot<T> = [PackedSharedValue<T>; 2];
type NodeShard = Mutex<SharedShard<GreenNode>>;
type TokenShard = Mutex<SharedShard<GreenToken>>;

trait SharedValue: Clone + PartialEq {
    type Data;

    fn as_ptr(&self) -> NonNull<Self::Data>;
    fn into_raw(self) -> NonNull<Self::Data>;

    /// # Safety
    ///
    /// `ptr` must have been returned by `Self::into_raw` and still own that reference.
    unsafe fn from_raw(ptr: NonNull<Self::Data>) -> Self;
}

impl SharedValue for GreenNode {
    type Data = GreenNodeData;

    fn as_ptr(&self) -> NonNull<Self::Data> {
        NonNull::from(&**self)
    }

    fn into_raw(self) -> NonNull<Self::Data> {
        GreenNode::into_raw(self)
    }

    unsafe fn from_raw(ptr: NonNull<Self::Data>) -> Self {
        // SAFETY: The caller upholds `SharedValue::from_raw`'s ownership contract.
        unsafe { GreenNode::from_raw(ptr) }
    }
}

impl SharedValue for GreenToken {
    type Data = GreenTokenData;

    fn as_ptr(&self) -> NonNull<Self::Data> {
        NonNull::from(&**self)
    }

    fn into_raw(self) -> NonNull<Self::Data> {
        GreenToken::into_raw(self)
    }

    unsafe fn from_raw(ptr: NonNull<Self::Data>) -> Self {
        // SAFETY: The caller upholds `SharedValue::from_raw`'s ownership contract.
        unsafe { GreenToken::from_raw(ptr) }
    }
}

const EMPTY_SHARED_VALUE: i32 = 0;
const WIDE_SHARED_VALUE: i32 = i32::MIN;

#[repr(transparent)]
struct PackedSharedValue<T: SharedValue> {
    offset: i32,
    _marker: PhantomData<T>,
}

impl<T: SharedValue> PackedSharedValue<T> {
    fn empty() -> Self {
        Self { offset: EMPTY_SHARED_VALUE, _marker: PhantomData }
    }

    fn is_empty(&self) -> bool {
        self.offset == EMPTY_SHARED_VALUE
    }

    fn is_wide(&self) -> bool {
        self.offset == WIDE_SHARED_VALUE
    }

    fn set_wide(&mut self) {
        debug_assert!(self.is_empty());
        self.offset = WIDE_SHARED_VALUE;
    }

    fn with_value<R>(&self, f: impl FnOnce(&T) -> R) -> Option<R> {
        if self.is_empty() || self.is_wide() {
            return None;
        }
        // SAFETY: A non-sentinel offset was created by `store`, which transferred one owner into
        // this stable boxed slot. The temporary owner is not dropped and cannot escape `f`.
        let owner = ManuallyDrop::new(unsafe { T::from_raw(self.ptr()) });
        Some(f(&owner))
    }

    fn store(&mut self, value: T) -> Result<(), T> {
        debug_assert!(self.is_empty());
        let base = self as *const Self as usize;
        let target = value.as_ptr().as_ptr().expose_provenance();
        let delta = target as i128 - base as i128;
        if delta % 4 != 0 {
            return Err(value);
        }
        let Ok(offset) = i32::try_from(delta / 4) else {
            return Err(value);
        };
        if matches!(offset, EMPTY_SHARED_VALUE | WIDE_SHARED_VALUE) {
            return Err(value);
        }
        _ = value.into_raw();
        self.offset = offset;
        Ok(())
    }

    fn clear(&mut self) {
        if !self.is_empty() && !self.is_wide() {
            // SAFETY: `store` transferred exactly one owner into this slot.
            drop(unsafe { T::from_raw(self.ptr()) });
        }
        self.offset = EMPTY_SHARED_VALUE;
    }

    /// # Safety
    ///
    /// The slot must contain a non-sentinel offset written by `store`.
    unsafe fn ptr(&self) -> NonNull<T::Data> {
        let base = self as *const Self as usize;
        let byte_offset = i64::from(self.offset) * 4;
        let address = base.wrapping_add(byte_offset as usize);
        // SAFETY: The stored offset names the live allocation whose owner is held by this slot.
        unsafe { NonNull::new_unchecked(ptr::with_exposed_provenance_mut(address)) }
    }
}

impl<T: SharedValue> Drop for PackedSharedValue<T> {
    fn drop(&mut self) {
        self.clear();
    }
}

impl<T: SharedValue> fmt::Debug for PackedSharedValue<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PackedSharedValue").field(&self.offset).finish()
    }
}

#[derive(Debug)]
struct SharedShard<T: SharedValue> {
    slots: Box<[SharedSlot<T>]>,
    wide: Vec<(usize, T)>,
}

impl<T: SharedValue> Default for SharedShard<T> {
    fn default() -> Self {
        SharedShard { slots: Box::new([]), wide: Vec::new() }
    }
}

impl<T: SharedValue> SharedShard<T> {
    #[cfg(test)]
    fn slot_count(&self) -> usize {
        self.slots.len() * 2
    }

    fn get(&self, hash: u64, mut matches: impl FnMut(&T) -> bool) -> Option<T> {
        if self.slots.is_empty() {
            return None;
        }
        let set = self.set_index(hash);
        (0..2).find_map(|way| {
            self.with_value(set, way, |value| matches(value).then(|| value.clone())).flatten()
        })
    }

    fn insert(&mut self, hash: u64, value: T, capacity: usize) -> T {
        if self.slots.is_empty() {
            self.slots = (0..capacity.div_ceil(2))
                .map(|_| std::array::from_fn(|_| PackedSharedValue::empty()))
                .collect::<Box<[_]>>();
        }
        let set = self.set_index(hash);
        for way in 0..2 {
            if let Some(cached) = self
                .with_value(set, way, |cached| (cached == &value).then(|| cached.clone()))
                .flatten()
            {
                return cached;
            }
        }
        let way = self.slots[set]
            .iter()
            .position(PackedSharedValue::is_empty)
            .unwrap_or((hash >> 32) as usize % 2);
        self.replace(set, way, value.clone());
        value
    }

    fn with_value<R>(&self, set: usize, way: usize, f: impl FnOnce(&T) -> R) -> Option<R> {
        let slot = &self.slots[set][way];
        if slot.is_wide() {
            self.wide.iter().find(|(key, _)| *key == set * 2 + way).map(|(_, value)| f(value))
        } else {
            slot.with_value(f)
        }
    }

    #[cfg(test)]
    fn value(&self, set: usize, way: usize) -> Option<T> {
        self.with_value(set, way, Clone::clone)
    }

    fn replace(&mut self, set: usize, way: usize, value: T) {
        let key = set * 2 + way;
        let slot = &mut self.slots[set][way];
        if slot.is_wide() {
            if let Some(index) = self.wide.iter().position(|(stored_key, _)| *stored_key == key) {
                self.wide.swap_remove(index);
            } else {
                debug_assert!(false, "wide cache slot must own a fallback value");
            }
            slot.offset = EMPTY_SHARED_VALUE;
        } else {
            slot.clear();
        }
        if let Err(value) = slot.store(value) {
            self.wide.push((key, value));
            slot.set_wide();
        }
    }

    fn set_index(&self, hash: u64) -> usize {
        (hash >> 8) as usize % self.slots.len()
    }
}

#[derive(Debug)]
struct NoHash<T>(T);

/// Interner for GreenTokens and GreenNodes.
#[derive(Default, Debug)]
pub struct NodeCache {
    nodes: HashMap<NoHash<GreenNode>, ()>,
    tokens: HashMap<NoHash<GreenToken>, ()>,
}

const SHARD_COUNT: usize = 256;
const NODE_CAPACITY_PER_SHARD: usize = 14 * 1024;
const TOKEN_CAPACITY_PER_SHARD: usize = 4 * 1024;
const MAX_SHARED_NODE_CHILDREN: usize = 1;
const MAX_SHARED_TOKEN_LEN: usize = 8;

/// A bounded, thread-safe interner for sharing immutable green trees across builders.
///
/// Individual shards use bounded two-way slots. A full slot overwrites one colliding entry, and
/// hash matches are confirmed with structural equality before a tree is reused.
#[derive(Debug)]
pub struct SharedNodeCache {
    nodes: Box<[NodeShard]>,
    tokens: Box<[TokenShard]>,
}

impl Default for SharedNodeCache {
    fn default() -> Self {
        let nodes = (0..SHARD_COUNT).map(|_| Mutex::new(SharedShard::default())).collect();
        let tokens = (0..SHARD_COUNT).map(|_| Mutex::new(SharedShard::default())).collect();
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
            *shard.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = SharedShard::default();
        }
        for shard in &self.tokens {
            *shard.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = SharedShard::default();
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
        shard.get(hash, |cached| node_matches(cached, kind, children))
    }

    fn insert_node(&self, hash: u64, node: GreenNode) -> GreenNode {
        let mut shard = self.nodes[hash as usize % SHARD_COUNT]
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        shard.insert(hash, node, NODE_CAPACITY_PER_SHARD)
    }

    fn token(&self, hash: u64, kind: SyntaxKind, text: &str) -> Option<GreenToken> {
        let shard = self.tokens[hash as usize % SHARD_COUNT]
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        shard.get(hash, |cached| cached.kind() == kind && cached.text() == text)
    }

    fn insert_token(&self, hash: u64, token: GreenToken) -> GreenToken {
        let mut shard = self.tokens[hash as usize % SHARD_COUNT]
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        shard.insert(hash, token, TOKEN_CAPACITY_PER_SHARD)
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
            match children_ref {
                [] => {}
                [(_, hash, _)] => hash.hash(&mut h),
                _ => {
                    for &(_, hash, _) in children_ref {
                        hash.hash(&mut h);
                    }
                }
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
        let repeated_first = cache.insert_node(0, GreenNode::new(SyntaxKind(1), []));

        assert!(!std::ptr::eq::<GreenNodeData>(&*first, &*second));
        assert!(std::ptr::eq::<GreenNodeData>(&*second, &*repeated));
        assert!(std::ptr::eq::<GreenNodeData>(&*first, &*repeated_first));
    }

    #[test]
    fn shared_cache_reserves_bounded_shards_on_first_use() {
        let cache = SharedNodeCache::default();
        cache.insert_node(0, GreenNode::new(SyntaxKind(1), []));
        cache.insert_token(0, GreenToken::new(SyntaxKind(1), "x"));

        assert_eq!(cache.nodes[0].lock().unwrap().slot_count(), NODE_CAPACITY_PER_SHARD);
        assert_eq!(cache.tokens[0].lock().unwrap().slot_count(), TOKEN_CAPACITY_PER_SHARD);
        assert_eq!(std::mem::size_of::<SharedSlot<GreenNode>>(), 2 * std::mem::size_of::<u32>());
    }

    #[test]
    fn shared_cache_wide_fallback_preserves_ownership() {
        let token = GreenToken::new(SyntaxKind(1), "fallback");
        let mut shard = SharedShard {
            slots: Box::new([[PackedSharedValue::empty(), PackedSharedValue::empty()]]),
            ..SharedShard::default()
        };
        shard.wide.push((0, token.clone()));
        shard.slots[0][0].set_wide();

        let cached = shard.value(0, 0).unwrap();
        assert!(std::ptr::eq::<GreenTokenData>(&*token, &*cached));
        drop(shard);
        assert_eq!(cached.text(), "fallback");
    }
}
