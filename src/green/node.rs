use std::{
    alloc::{self, Layout},
    borrow::Borrow,
    fmt,
    hash::{Hash, Hasher},
    iter::{self, FusedIterator},
    mem::{self, ManuallyDrop},
    num::NonZeroUsize,
    ops, ptr, slice,
    sync::atomic::{
        AtomicUsize,
        Ordering::{Acquire, Relaxed, Release},
    },
};

use countme::Count;
use memoffset::offset_of;

use crate::{
    green::{GreenElement, GreenElementRef, SyntaxKind},
    utility_types::static_assert,
    GreenToken, GreenTokenData, NodeOrToken, TextRange, TextSize,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct GreenNodeHead {
    kind: SyntaxKind,
    text_len: TextSize,
    _c: Count<GreenNode>,
}

/// An owning node-or-token pointer.
///
/// The pointer's low bit distinguishes tokens from nodes. Relative offsets are
/// checkpointed once per four children in the containing node.
#[repr(transparent)]
pub(crate) struct GreenChild {
    ptr: ptr::NonNull<()>,
}
#[cfg(target_pointer_width = "64")]
static_assert!(mem::size_of::<GreenChild>() == 8);
static_assert!(mem::align_of::<GreenNodeData>() >= 2);
static_assert!(mem::align_of::<GreenTokenData>() >= 2);

const CHILDREN_PER_CHECKPOINT: usize = 4;
const MAX_REFCOUNT: usize = isize::MAX as usize;

#[repr(C)]
pub struct GreenNodeData {
    header: GreenNodeHead,
    child_count: u32,
    children: [GreenChild; 0],
}

#[repr(C)]
struct GreenNodeAllocation {
    count: AtomicUsize,
    data: GreenNodeData,
}

struct GreenNodeAllocGuard {
    allocation: ptr::NonNull<GreenNodeAllocation>,
    child_count: usize,
    initialized_children: usize,
}

impl Drop for GreenNodeAllocGuard {
    fn drop(&mut self) {
        // SAFETY: The guard owns the unpublished allocation and tracks exactly how many
        // children were initialized before construction unwound.
        unsafe {
            let child_ptr =
                ptr::addr_of_mut!((*self.allocation.as_ptr()).data.children).cast::<GreenChild>();
            for index in 0..self.initialized_children {
                ptr::drop_in_place(child_ptr.add(index));
            }
            alloc::dealloc(self.allocation.cast().as_ptr(), allocation_layout(self.child_count));
        }
    }
}

impl PartialEq for GreenNodeData {
    fn eq(&self, other: &Self) -> bool {
        self.header() == other.header()
            && self.children_with_offsets().eq(other.children_with_offsets())
    }
}

/// Internal node in the immutable tree.
/// It has other nodes and tokens as children.
#[repr(transparent)]
pub struct GreenNode {
    ptr: ptr::NonNull<GreenNodeData>,
}

impl ToOwned for GreenNodeData {
    type Owned = GreenNode;

    #[inline]
    fn to_owned(&self) -> GreenNode {
        unsafe {
            let green = GreenNode::from_raw(ptr::NonNull::from(self));
            let green = ManuallyDrop::new(green);
            GreenNode::clone(&green)
        }
    }
}

impl Borrow<GreenNodeData> for GreenNode {
    #[inline]
    fn borrow(&self) -> &GreenNodeData {
        &*self
    }
}

impl fmt::Debug for GreenNodeData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GreenNode")
            .field("kind", &self.kind())
            .field("text_len", &self.text_len())
            .field("n_children", &self.children().len())
            .finish()
    }
}

impl fmt::Debug for GreenNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data: &GreenNodeData = &*self;
        fmt::Debug::fmt(data, f)
    }
}

impl fmt::Display for GreenNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data: &GreenNodeData = &*self;
        fmt::Display::fmt(data, f)
    }
}

impl fmt::Display for GreenNodeData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for child in self.children() {
            write!(f, "{}", child)?;
        }
        Ok(())
    }
}

impl GreenNodeData {
    #[inline]
    fn header(&self) -> &GreenNodeHead {
        &self.header
    }

    #[inline]
    fn slice(&self) -> &[GreenChild] {
        // SAFETY: The allocation stores exactly `child_count` initialized children here.
        unsafe { slice::from_raw_parts(self.children.as_ptr(), self.child_count as usize) }
    }

    #[inline]
    fn checkpoints(&self) -> &[TextSize] {
        let len = (self.child_count as usize).saturating_sub(1) / CHILDREN_PER_CHECKPOINT;
        let ptr = self.children.as_ptr().wrapping_add(self.child_count as usize).cast();
        // SAFETY: Construction writes one checkpoint after the child tail for every block after
        // the first. The first block always starts at zero.
        unsafe { slice::from_raw_parts(ptr, len) }
    }

    #[inline]
    fn block_offset(&self, block: usize) -> TextSize {
        if block == 0 {
            0.into()
        } else {
            self.checkpoints()[block - 1]
        }
    }

    #[inline]
    pub(crate) fn child_count(&self) -> usize {
        self.child_count as usize
    }

    #[inline]
    pub(crate) fn child(&self, index: usize) -> Option<GreenElementRef<'_>> {
        self.slice().get(index).map(GreenChild::as_ref)
    }

    #[inline]
    pub(crate) fn child_with_offset(&self, index: usize) -> GreenChildRef<'_> {
        let block = index / CHILDREN_PER_CHECKPOINT;
        let block_start = block * CHILDREN_PER_CHECKPOINT;
        let mut rel_offset = self.block_offset(block);
        for child in &self.slice()[block_start..index] {
            rel_offset += child.as_ref().text_len();
        }
        GreenChildRef { element: self.slice()[index].as_ref(), rel_offset }
    }

    #[inline]
    pub(crate) fn children_with_offsets(&self) -> GreenChildren<'_> {
        GreenChildren::new(self)
    }

    /// Kind of this node.
    #[inline]
    pub fn kind(&self) -> SyntaxKind {
        self.header().kind
    }

    /// Returns the length of the text covered by this node.
    #[inline]
    pub fn text_len(&self) -> TextSize {
        self.header().text_len
    }

    /// Children of this node.
    #[inline]
    pub fn children(&self) -> Children<'_> {
        Children { raw: self.children_with_offsets() }
    }

    pub(crate) fn child_at_range(
        &self,
        rel_range: TextRange,
    ) -> Option<(usize, TextSize, GreenElementRef<'_>)> {
        if self.child_count() == 0 {
            return None;
        }
        let block = self.checkpoints().partition_point(|&offset| offset <= rel_range.start());
        let start = block * CHILDREN_PER_CHECKPOINT;
        let end = (start + CHILDREN_PER_CHECKPOINT).min(self.slice().len());
        let mut rel_offset = self.block_offset(block);
        let mut candidate = start.checked_sub(1).map(|index| {
            let child = self.child_with_offset(index);
            (index, child.rel_offset, child.element)
        });
        for index in start..end {
            let element = self.slice()[index].as_ref();
            let child_range = TextRange::at(rel_offset, element.text_len());
            let current = (index, rel_offset, element);
            rel_offset += element.text_len();
            match TextRange::ordering(child_range, rel_range) {
                std::cmp::Ordering::Less => candidate = Some(current),
                std::cmp::Ordering::Equal => {
                    candidate = Some(current);
                    break;
                }
                std::cmp::Ordering::Greater => {
                    candidate.get_or_insert(current);
                    break;
                }
            }
        }
        candidate.filter(|&(_, offset, element)| {
            TextRange::at(offset, element.text_len()).contains_range(rel_range)
        })
    }

    #[cfg(test)]
    pub(crate) fn child_offset(&self, index: usize) -> TextSize {
        self.child_with_offset(index).rel_offset
    }
    #[must_use]
    pub fn replace_child(&self, index: usize, new_child: GreenElement) -> GreenNode {
        let mut replacement = Some(new_child);
        let children = self.children().enumerate().map(|(i, child)| {
            if i == index {
                replacement.take().unwrap()
            } else {
                child.to_owned()
            }
        });
        GreenNode::new(self.kind(), children)
    }
    #[must_use]
    pub fn insert_child(&self, index: usize, new_child: GreenElement) -> GreenNode {
        // https://github.com/rust-lang/rust/issues/34433
        self.splice_children(index..index, iter::once(new_child))
    }
    #[must_use]
    pub fn remove_child(&self, index: usize) -> GreenNode {
        self.splice_children(index..=index, iter::empty())
    }
    #[must_use]
    pub fn splice_children<R, I>(&self, range: R, replace_with: I) -> GreenNode
    where
        R: ops::RangeBounds<usize>,
        I: IntoIterator<Item = GreenElement>,
    {
        let mut children: Vec<_> = self.children().map(|it| it.to_owned()).collect();
        children.splice(range, replace_with);
        GreenNode::new(self.kind(), children)
    }
}

impl ops::Deref for GreenNode {
    type Target = GreenNodeData;

    #[inline]
    fn deref(&self) -> &GreenNodeData {
        // SAFETY: `GreenNode` owns one strong reference to this immutable allocation.
        unsafe { self.ptr.as_ref() }
    }
}

fn allocation_layout(child_count: usize) -> Layout {
    let checkpoints = child_count.saturating_sub(1) / CHILDREN_PER_CHECKPOINT;
    let children_offset =
        offset_of!(GreenNodeAllocation, data) + offset_of!(GreenNodeData, children);
    let usable_size = children_offset
        .checked_add(mem::size_of::<GreenChild>().checked_mul(child_count).unwrap())
        .and_then(|size| {
            size.checked_add(mem::size_of::<TextSize>().checked_mul(checkpoints).unwrap())
        })
        .expect("green node allocation size overflows");
    let align = mem::align_of::<GreenNodeAllocation>();
    let size = usable_size.checked_add(align - 1).unwrap() & !(align - 1);
    Layout::from_size_align(size, align).expect("invalid green node allocation layout")
}

#[inline]
unsafe fn allocation_ptr(data: ptr::NonNull<GreenNodeData>) -> ptr::NonNull<GreenNodeAllocation> {
    unsafe {
        data.cast::<u8>().sub(offset_of!(GreenNodeAllocation, data)).cast::<GreenNodeAllocation>()
    }
}

impl Clone for GreenNode {
    #[inline]
    fn clone(&self) -> Self {
        // SAFETY: `self` keeps the allocation alive.
        let allocation = unsafe { allocation_ptr(self.ptr).as_ref() };
        let old_size = allocation.count.fetch_add(1, Relaxed);
        if old_size > MAX_REFCOUNT {
            std::process::abort();
        }
        GreenNode { ptr: self.ptr }
    }
}

impl Drop for GreenNode {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: `self` owns one strong reference.
        let allocation = unsafe { allocation_ptr(self.ptr) };
        if unsafe { allocation.as_ref() }.count.fetch_sub(1, Release) != 1 {
            return;
        }
        // SAFETY: This was the final strong reference.
        unsafe { self.drop_slow(allocation) };
    }
}

impl GreenNode {
    #[inline(never)]
    unsafe fn drop_slow(&mut self, allocation: ptr::NonNull<GreenNodeAllocation>) {
        unsafe { allocation.as_ref() }.count.load(Acquire);

        let child_count = self.child_count as usize;
        for child in self.slice() {
            // SAFETY: This is the final strong reference, so every initialized child
            // can be dropped exactly once.
            unsafe { ptr::drop_in_place(child as *const GreenChild as *mut GreenChild) };
        }
        // SAFETY: The header was initialized during construction and is dropped once.
        unsafe { ptr::drop_in_place(ptr::addr_of_mut!((*self.ptr.as_ptr()).header)) };
        // SAFETY: `allocation` was allocated with this exact layout and all fields are dropped.
        unsafe { alloc::dealloc(allocation.cast().as_ptr(), allocation_layout(child_count)) };
    }
}

// SAFETY: Green nodes are immutable and their reference count is atomic.
unsafe impl Send for GreenNode {}
// SAFETY: Green nodes are immutable and their reference count is atomic.
unsafe impl Sync for GreenNode {}

impl PartialEq for GreenNode {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(&**self, &**other) || **self == **other
    }
}

impl Eq for GreenNode {}

impl Hash for GreenNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

impl Hash for GreenNodeData {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.header().hash(state);
        for child in self.children_with_offsets() {
            hash_element(child.element, child.rel_offset, state);
        }
    }
}

impl GreenNode {
    /// Creates new Node.
    #[inline]
    pub fn new<I>(kind: SyntaxKind, children: I) -> GreenNode
    where
        I: IntoIterator<Item = GreenElement>,
        I::IntoIter: ExactSizeIterator,
    {
        let mut children = children.into_iter();
        let child_count = children.len();
        let stored_child_count =
            u32::try_from(child_count).expect("green node child count exceeds u32::MAX");
        let layout = allocation_layout(child_count);
        // SAFETY: `layout` is non-zero and valid.
        let buffer = unsafe { alloc::alloc(layout) };
        if buffer.is_null() {
            alloc::handle_alloc_error(layout);
        }
        let allocation = buffer.cast::<GreenNodeAllocation>();
        // SAFETY: `alloc::alloc` returned a non-null pointer aligned for this layout.
        let allocation = unsafe { ptr::NonNull::new_unchecked(allocation) };
        let mut guard = GreenNodeAllocGuard { allocation, child_count, initialized_children: 0 };
        // SAFETY: The allocation is valid and properly aligned for all writes below.
        unsafe {
            ptr::write(ptr::addr_of_mut!((*allocation.as_ptr()).count), AtomicUsize::new(1));
            ptr::write(
                ptr::addr_of_mut!((*allocation.as_ptr()).data.child_count),
                stored_child_count,
            );
        }

        let mut text_len: TextSize = 0.into();
        // SAFETY: `children` is the start of the packed child tail.
        let child_ptr =
            unsafe { ptr::addr_of_mut!((*allocation.as_ptr()).data.children).cast::<GreenChild>() };
        // SAFETY: The checkpoint tail immediately follows all children and remains aligned.
        let checkpoint_ptr = unsafe { child_ptr.add(child_count).cast::<TextSize>() };
        for index in 0..child_count {
            if index != 0 && index % CHILDREN_PER_CHECKPOINT == 0 {
                // SAFETY: The layout reserves one checkpoint for every block after the first.
                unsafe {
                    ptr::write(checkpoint_ptr.add(index / CHILDREN_PER_CHECKPOINT - 1), text_len)
                };
            }
            let element = children.next().expect("ExactSizeIterator over-reported length");
            text_len += element.text_len();
            // SAFETY: Each child slot is initialized exactly once.
            unsafe { ptr::write(child_ptr.add(index), GreenChild::from_element(element)) };
            guard.initialized_children += 1;
        }
        assert!(children.next().is_none(), "ExactSizeIterator under-reported length");
        // SAFETY: Header initialization completes the allocation before it is published.
        unsafe {
            ptr::write(
                ptr::addr_of_mut!((*allocation.as_ptr()).data.header),
                GreenNodeHead { kind, text_len, _c: Count::new() },
            );
        }
        mem::forget(guard);

        // SAFETY: `allocation` is non-null and `data` is fully initialized.
        let ptr =
            unsafe { ptr::NonNull::new_unchecked(ptr::addr_of_mut!((*allocation.as_ptr()).data)) };
        GreenNode { ptr }
    }

    #[inline]
    pub(crate) fn into_raw(this: GreenNode) -> ptr::NonNull<GreenNodeData> {
        ManuallyDrop::new(this).ptr
    }

    #[inline]
    pub(crate) unsafe fn from_raw(ptr: ptr::NonNull<GreenNodeData>) -> GreenNode {
        GreenNode { ptr }
    }
}

impl GreenChild {
    const TOKEN_TAG: usize = 1;

    fn from_node(node: GreenNode) -> GreenChild {
        let ptr = GreenNode::into_raw(node).cast();
        debug_assert_eq!(ptr.addr().get() & Self::TOKEN_TAG, 0);
        GreenChild { ptr }
    }

    fn from_token(token: GreenToken) -> GreenChild {
        let ptr = GreenToken::into_raw(token).cast();
        debug_assert_eq!(ptr.addr().get() & Self::TOKEN_TAG, 0);
        let ptr = ptr.map_addr(|addr| NonZeroUsize::new(addr.get() | Self::TOKEN_TAG).unwrap());
        GreenChild { ptr }
    }

    fn from_element(element: GreenElement) -> GreenChild {
        match element {
            NodeOrToken::Node(node) => GreenChild::from_node(node),
            NodeOrToken::Token(token) => GreenChild::from_token(token),
        }
    }

    fn ptr(&self) -> ptr::NonNull<()> {
        self.ptr
    }

    fn is_token(ptr: ptr::NonNull<()>) -> bool {
        ptr.addr().get() & Self::TOKEN_TAG != 0
    }

    fn untagged<T>(ptr: ptr::NonNull<()>) -> ptr::NonNull<T> {
        ptr.map_addr(|addr| NonZeroUsize::new(addr.get() & !Self::TOKEN_TAG).unwrap()).cast()
    }

    #[inline]
    pub(crate) fn as_ref(&self) -> GreenElementRef<'_> {
        let ptr = self.ptr();
        if Self::is_token(ptr) {
            let ptr = Self::untagged::<GreenTokenData>(ptr);
            // SAFETY: `from_token` stores one owned token pointer, and `self` keeps it alive.
            NodeOrToken::Token(unsafe { ptr.as_ref() })
        } else {
            let ptr = Self::untagged::<GreenNodeData>(ptr);
            // SAFETY: `from_node` stores one owned node pointer, and `self` keeps it alive.
            NodeOrToken::Node(unsafe { ptr.as_ref() })
        }
    }
}

impl Drop for GreenChild {
    fn drop(&mut self) {
        let ptr = self.ptr();
        if Self::is_token(ptr) {
            // SAFETY: `from_token` transferred exactly one owned token reference into this child.
            drop(unsafe { GreenToken::from_raw(Self::untagged(ptr)) });
        } else {
            // SAFETY: `from_node` transferred exactly one owned node reference into this child.
            drop(unsafe { GreenNode::from_raw(Self::untagged(ptr)) });
        }
    }
}

fn hash_element<H: Hasher>(element: GreenElementRef<'_>, rel_offset: TextSize, state: &mut H) {
    rel_offset.hash(state);
    match element {
        NodeOrToken::Node(node) => {
            false.hash(state);
            node.kind().hash(state);
            node.text_len().hash(state);
            node.children().len().hash(state);
            for child in node.children_with_offsets() {
                hash_element(child.element, child.rel_offset, state);
            }
        }
        NodeOrToken::Token(token) => {
            true.hash(state);
            token.kind().hash(state);
            token.text().hash(state);
        }
    }
}

// SAFETY: `GreenChild` owns an immutable `GreenNode` or `GreenToken`, both Send and Sync.
unsafe impl Send for GreenChild {}
// SAFETY: `GreenChild` owns an immutable `GreenNode` or `GreenToken`, both Send and Sync.
unsafe impl Sync for GreenChild {}

#[derive(Clone, Copy)]
pub(crate) struct GreenChildRef<'a> {
    pub(crate) element: GreenElementRef<'a>,
    pub(crate) rel_offset: TextSize,
}

impl PartialEq for GreenChildRef<'_> {
    fn eq(&self, other: &Self) -> bool {
        if self.rel_offset != other.rel_offset {
            return false;
        }
        match (self.element, other.element) {
            (NodeOrToken::Node(left), NodeOrToken::Node(right)) => {
                ptr::eq(left, right) || left == right
            }
            (NodeOrToken::Token(left), NodeOrToken::Token(right)) => {
                ptr::eq(left, right) || left == right
            }
            _ => false,
        }
    }
}

impl<'a> GreenChildRef<'a> {
    #[inline]
    pub(crate) fn as_ref(self) -> GreenElementRef<'a> {
        self.element
    }

    #[inline]
    pub(crate) fn rel_offset(self) -> TextSize {
        self.rel_offset
    }
}

#[derive(Clone)]
pub(crate) struct GreenChildren<'a> {
    children: &'a [GreenChild],
    checkpoints: &'a [TextSize],
    front: usize,
    back: usize,
    front_offset: TextSize,
    back_offset: TextSize,
}

impl<'a> GreenChildren<'a> {
    #[inline]
    fn new(node: &'a GreenNodeData) -> Self {
        GreenChildren {
            children: node.slice(),
            checkpoints: node.checkpoints(),
            front: 0,
            back: node.slice().len(),
            front_offset: 0.into(),
            back_offset: node.text_len(),
        }
    }

    #[inline]
    fn child_at(&self, index: usize) -> GreenChildRef<'a> {
        let block = index / CHILDREN_PER_CHECKPOINT;
        let block_start = block * CHILDREN_PER_CHECKPOINT;
        let mut rel_offset = if block == 0 { 0.into() } else { self.checkpoints[block - 1] };
        for child in &self.children[block_start..index] {
            rel_offset += child.as_ref().text_len();
        }
        GreenChildRef { element: self.children[index].as_ref(), rel_offset }
    }
}

impl ExactSizeIterator for GreenChildren<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.back - self.front
    }
}

impl<'a> Iterator for GreenChildren<'a> {
    type Item = GreenChildRef<'a>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.front == self.back {
            return None;
        }
        let element = self.children[self.front].as_ref();
        let child = GreenChildRef { element, rel_offset: self.front_offset };
        self.front += 1;
        self.front_offset += element.text_len();
        Some(child)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if n >= self.len() {
            self.front = self.back;
            return None;
        }
        let index = self.front + n;
        let child = self.child_at(index);
        self.front = index + 1;
        self.front_offset = child.rel_offset + child.element.text_len();
        Some(child)
    }
}

impl DoubleEndedIterator for GreenChildren<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front == self.back {
            return None;
        }
        self.back -= 1;
        let element = self.children[self.back].as_ref();
        self.back_offset -= element.text_len();
        Some(GreenChildRef { element, rel_offset: self.back_offset })
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        if n >= self.len() {
            self.back = self.front;
            return None;
        }
        let index = self.back - n - 1;
        let child = self.child_at(index);
        self.back = index;
        self.back_offset = child.rel_offset;
        Some(child)
    }
}

impl FusedIterator for GreenChildren<'_> {}

#[derive(Clone)]
pub struct Children<'a> {
    pub(crate) raw: GreenChildren<'a>,
}

impl fmt::Debug for Children<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

impl ExactSizeIterator for Children<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.raw.len()
    }
}

impl<'a> Iterator for Children<'a> {
    type Item = GreenElementRef<'a>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.raw.next().map(GreenChildRef::as_ref)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.raw.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.raw.count()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.raw.nth(n).map(GreenChildRef::as_ref)
    }
}

impl DoubleEndedIterator for Children<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.raw.next_back().map(GreenChildRef::as_ref)
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.raw.nth_back(n).map(GreenChildRef::as_ref)
    }
}

impl FusedIterator for Children<'_> {}
