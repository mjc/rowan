#[cfg(not(target_pointer_width = "64"))]
use std::sync::atomic::AtomicUsize;
use std::{
    alloc::Layout,
    borrow::Borrow,
    fmt,
    hash::{Hash, Hasher},
    iter::{self, FusedIterator},
    mem::{self, ManuallyDrop},
    num::NonZeroUsize,
    ops, ptr, slice,
    sync::atomic::{
        AtomicU32,
        Ordering::{Acquire, Relaxed, Release},
    },
};

use countme::Count;
use memoffset::offset_of;

use crate::{
    green::{allocator, GreenElement, GreenElementRef, SyntaxKind},
    utility_types::static_assert,
    GreenToken, GreenTokenData, NodeOrToken, TextRange, TextSize,
};

#[cfg(not(target_pointer_width = "64"))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct GreenNodeHead {
    kind: SyntaxKind,
    child_count: u16,
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
/// A child pointer compressed into a signed offset from its parent node.
///
/// The low bit distinguishes tokens from nodes. The remaining bits store a signed offset in
/// four-byte units, covering the 8 GiB range centered on the parent; green nodes and tokens are
/// both at least four-byte aligned.
#[cfg(target_pointer_width = "64")]
#[derive(Clone, Copy)]
#[repr(transparent)]
pub(super) struct PackedGreenChild(u32);
#[cfg(target_pointer_width = "64")]
static_assert!(mem::size_of::<GreenChild>() == 8);
#[cfg(target_pointer_width = "64")]
static_assert!(mem::size_of::<PackedGreenChild>() == 4);
static_assert!(mem::align_of::<GreenNodeData>() >= 4);
static_assert!(mem::align_of::<GreenTokenData>() >= 4);

const CHILDREN_PER_CHECKPOINT: usize = 4;
#[cfg(not(target_pointer_width = "64"))]
const WIDE_CHILD_COUNT: u16 = u16::MAX;
#[cfg(target_pointer_width = "64")]
const MAX_REFCOUNT: u32 = i32::MAX as u32;
#[cfg(not(target_pointer_width = "64"))]
const MAX_REFCOUNT: usize = isize::MAX as usize;
#[cfg(target_pointer_width = "64")]
const PACKED_KIND_BITS: u32 = 10;
#[cfg(target_pointer_width = "64")]
const PACKED_CHILD_COUNT_BITS: u32 = 8;
#[cfg(target_pointer_width = "64")]
const PACKED_KIND_LIMIT: u32 = (1 << PACKED_KIND_BITS) - 1;
#[cfg(target_pointer_width = "64")]
const PACKED_CHILD_COUNT_LIMIT: u32 = (1 << PACKED_CHILD_COUNT_BITS) - 1;
#[cfg(target_pointer_width = "64")]
const PACKED_TEXT_LEN_LIMIT: u32 = (1 << (32 - PACKED_KIND_BITS - PACKED_CHILD_COUNT_BITS)) - 1;
#[cfg(target_pointer_width = "64")]
const WIDE_PACKED_HEAD: u32 = u32::MAX;

#[repr(C)]
pub struct GreenNodeData {
    #[cfg(not(target_pointer_width = "64"))]
    header: GreenNodeHead,
    #[cfg(target_pointer_width = "64")]
    packed_head: u32,
    #[cfg(target_pointer_width = "64")]
    _c: Count<GreenNode>,
    children: [u8; 0],
}

#[cfg(target_pointer_width = "64")]
static_assert!(mem::size_of::<GreenNodeData>() == 4);

#[cfg(target_pointer_width = "64")]
#[repr(C)]
struct WideGreenNodeHead {
    child_count: u32,
    text_len: TextSize,
    kind: SyntaxKind,
    _padding: [u8; 6],
}

#[cfg(target_pointer_width = "64")]
static_assert!(mem::size_of::<WideGreenNodeHead>() == 16);

#[cfg(target_pointer_width = "64")]
type GreenNodeRefCount = AtomicU32;
#[cfg(not(target_pointer_width = "64"))]
type GreenNodeRefCount = AtomicUsize;

#[repr(C)]
struct GreenNodeAllocation {
    count: GreenNodeRefCount,
    data: GreenNodeData,
}

struct GreenNodeAllocGuard {
    allocation: ptr::NonNull<GreenNodeAllocation>,
    child_count: usize,
    wide: bool,
    initialized_children: usize,
}

impl Drop for GreenNodeAllocGuard {
    fn drop(&mut self) {
        // SAFETY: The guard owns the unpublished allocation and tracks exactly how many
        // children were initialized before construction unwound.
        unsafe {
            #[cfg(target_pointer_width = "64")]
            if !self.wide {
                let parent = ptr::NonNull::new_unchecked(ptr::addr_of_mut!(
                    (*self.allocation.as_ptr()).data
                ));
                let child_ptr = allocation_packed_child_ptr(self.allocation);
                for index in 0..self.initialized_children {
                    child_ptr.add(index).read().drop_owned(parent);
                }
            } else {
                let child_ptr = allocation_child_ptr(self.allocation, self.child_count, self.wide);
                for index in 0..self.initialized_children {
                    ptr::drop_in_place(child_ptr.add(index));
                }
            }
            #[cfg(not(target_pointer_width = "64"))]
            {
                let child_ptr = allocation_child_ptr(self.allocation, self.child_count, self.wide);
                for index in 0..self.initialized_children {
                    ptr::drop_in_place(child_ptr.add(index));
                }
            }
            allocator::deallocate(
                self.allocation.cast(),
                allocation_layout(self.child_count, self.wide),
            );
        }
    }
}

impl PartialEq for GreenNodeData {
    fn eq(&self, other: &Self) -> bool {
        self.kind() == other.kind()
            && self.text_len() == other.text_len()
            && self.child_count() == other.child_count()
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
    #[cfg(not(target_pointer_width = "64"))]
    #[inline]
    fn header(&self) -> &GreenNodeHead {
        &self.header
    }

    #[cfg(target_pointer_width = "64")]
    #[inline]
    fn is_wide(&self) -> bool {
        self.packed_head == WIDE_PACKED_HEAD
    }

    #[cfg(target_pointer_width = "64")]
    #[cold]
    #[inline(never)]
    fn wide_head(&self) -> &WideGreenNodeHead {
        // SAFETY: `is_wide()` means construction reserved and initialized this header.
        unsafe { &*self.children.as_ptr().cast::<WideGreenNodeHead>() }
    }

    #[cfg(target_pointer_width = "64")]
    #[inline]
    fn packed_child_ptr(&self) -> *const PackedGreenChild {
        self.children.as_ptr().cast()
    }

    #[inline]
    fn wide_child_ptr(&self) -> *const GreenChild {
        #[cfg(target_pointer_width = "64")]
        {
            self.children.as_ptr().cast::<GreenChild>().wrapping_add(2)
        }
        #[cfg(not(target_pointer_width = "64"))]
        {
            self.children
                .as_ptr()
                .cast::<GreenChild>()
                .wrapping_add(usize::from(self.header.child_count == WIDE_CHILD_COUNT))
        }
    }

    #[inline]
    fn child_ref(&self, index: usize) -> GreenElementRef<'_> {
        #[cfg(target_pointer_width = "64")]
        if !self.is_wide() {
            // SAFETY: Packed nodes store exactly `child_count()` initialized compact children.
            return unsafe {
                (*self.packed_child_ptr().add(index)).as_ref(ptr::NonNull::from(self))
            };
        }

        // SAFETY: Wide nodes store exactly `child_count()` initialized owning child pointers.
        unsafe { (*self.wide_child_ptr().add(index)).as_ref() }
    }

    #[inline]
    fn checkpoints_with_count(&self, child_count: usize) -> &[TextSize] {
        let len = child_count.saturating_sub(1) / CHILDREN_PER_CHECKPOINT;
        #[cfg(target_pointer_width = "64")]
        let ptr = if self.is_wide() {
            self.wide_child_ptr().wrapping_add(child_count).cast()
        } else {
            self.packed_child_ptr().wrapping_add(child_count).cast()
        };
        #[cfg(not(target_pointer_width = "64"))]
        let ptr = self.wide_child_ptr().wrapping_add(child_count).cast();
        // SAFETY: Construction writes one checkpoint after the child tail for every block after
        // the first. The first block always starts at zero.
        unsafe { slice::from_raw_parts(ptr, len) }
    }

    #[inline]
    fn block_offset(checkpoints: &[TextSize], block: usize) -> TextSize {
        if block == 0 {
            0.into()
        } else {
            checkpoints[block - 1]
        }
    }

    #[inline]
    pub(crate) fn child_count(&self) -> usize {
        #[cfg(target_pointer_width = "64")]
        {
            if !self.is_wide() {
                return ((self.packed_head >> PACKED_KIND_BITS) & PACKED_CHILD_COUNT_LIMIT)
                    as usize;
            }
            self.wide_head().child_count as usize
        }
        #[cfg(not(target_pointer_width = "64"))]
        {
            if self.header.child_count != WIDE_CHILD_COUNT {
                self.header.child_count as usize
            } else {
                self.wide_child_count()
            }
        }
    }

    #[cfg(not(target_pointer_width = "64"))]
    #[cold]
    #[inline(never)]
    fn wide_child_count(&self) -> usize {
        #[cfg(not(target_pointer_width = "64"))]
        {
            // SAFETY: Wide nodes store their u32 count in the aligned slot immediately
            // preceding the child tail.
            unsafe { self.children.as_ptr().cast::<u32>().read() as usize }
        }
    }

    #[inline]
    pub(crate) fn child(&self, index: usize) -> Option<GreenElementRef<'_>> {
        let child_count = self.child_count();
        (index < child_count).then(|| self.child_ref(index))
    }

    #[inline]
    pub(crate) fn children_with_offsets(&self) -> GreenChildren<'_> {
        GreenChildren::new(self)
    }

    /// Kind of this node.
    #[inline]
    pub fn kind(&self) -> SyntaxKind {
        #[cfg(target_pointer_width = "64")]
        {
            if !self.is_wide() {
                return SyntaxKind((self.packed_head & PACKED_KIND_LIMIT) as u16);
            }
            self.wide_head().kind
        }
        #[cfg(not(target_pointer_width = "64"))]
        {
            self.header().kind
        }
    }

    /// Returns the length of the text covered by this node.
    #[inline]
    pub fn text_len(&self) -> TextSize {
        #[cfg(target_pointer_width = "64")]
        {
            if !self.is_wide() {
                return (self.packed_head >> (PACKED_KIND_BITS + PACKED_CHILD_COUNT_BITS)).into();
            }
            self.wide_head().text_len
        }
        #[cfg(not(target_pointer_width = "64"))]
        {
            self.header().text_len
        }
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
        let child_count = self.child_count();
        if child_count == 0 {
            return None;
        }
        let checkpoints = self.checkpoints_with_count(child_count);
        let block = checkpoints.partition_point(|&offset| offset <= rel_range.start());
        let start = block * CHILDREN_PER_CHECKPOINT;
        let end = (start + CHILDREN_PER_CHECKPOINT).min(child_count);
        let mut rel_offset = Self::block_offset(checkpoints, block);
        let mut candidate = start.checked_sub(1).map(|index| {
            let previous_block = index / CHILDREN_PER_CHECKPOINT;
            let previous_block_start = previous_block * CHILDREN_PER_CHECKPOINT;
            let mut previous_offset = Self::block_offset(checkpoints, previous_block);
            for child_index in previous_block_start..index {
                previous_offset += self.child_ref(child_index).text_len();
            }
            (index, previous_offset, self.child_ref(index))
        });
        for index in start..end {
            let element = self.child_ref(index);
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
        self.children_with_offsets().nth(index).unwrap().rel_offset
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

pub(super) fn allocation_layout(child_count: usize, wide: bool) -> Layout {
    let checkpoints = child_count.saturating_sub(1) / CHILDREN_PER_CHECKPOINT;
    let children_offset =
        offset_of!(GreenNodeAllocation, data) + offset_of!(GreenNodeData, children);
    let wide_count_size = mem::size_of::<GreenChild>() * wide_count_slots(child_count, wide);
    #[cfg(target_pointer_width = "64")]
    let child_size =
        if wide { mem::size_of::<GreenChild>() } else { mem::size_of::<PackedGreenChild>() };
    #[cfg(not(target_pointer_width = "64"))]
    let child_size = mem::size_of::<GreenChild>();
    let children_size =
        child_size.checked_mul(child_count).expect("green node child allocation size overflows");
    let checkpoints_size = mem::size_of::<TextSize>()
        .checked_mul(checkpoints)
        .expect("green node checkpoint allocation size overflows");
    let usable_size = children_offset
        .checked_add(wide_count_size)
        .and_then(|size| size.checked_add(children_size))
        .and_then(|size| size.checked_add(checkpoints_size))
        .expect("green node allocation size overflows");
    #[cfg(target_pointer_width = "64")]
    let align =
        if wide { mem::align_of::<GreenChild>() } else { mem::align_of::<GreenNodeAllocation>() };
    #[cfg(not(target_pointer_width = "64"))]
    let align = mem::align_of::<GreenNodeAllocation>();
    let size = usable_size.checked_add(align - 1).unwrap() & !(align - 1);
    Layout::from_size_align(size, align).expect("invalid green node allocation layout")
}

#[inline]
const fn wide_count_slots(child_count: usize, wide: bool) -> usize {
    #[cfg(target_pointer_width = "64")]
    {
        let _ = child_count;
        if wide {
            2
        } else {
            0
        }
    }
    #[cfg(not(target_pointer_width = "64"))]
    {
        usize::from(child_count >= WIDE_CHILD_COUNT as usize)
    }
}

unsafe fn allocation_child_ptr(
    allocation: ptr::NonNull<GreenNodeAllocation>,
    child_count: usize,
    wide: bool,
) -> *mut GreenChild {
    unsafe {
        ptr::addr_of_mut!((*allocation.as_ptr()).data.children)
            .cast::<GreenChild>()
            .add(wide_count_slots(child_count, wide))
    }
}

#[cfg(target_pointer_width = "64")]
unsafe fn allocation_packed_child_ptr(
    allocation: ptr::NonNull<GreenNodeAllocation>,
) -> *mut PackedGreenChild {
    unsafe { ptr::addr_of_mut!((*allocation.as_ptr()).data.children).cast() }
}

#[cfg(target_pointer_width = "64")]
unsafe fn promote_to_wide(
    allocation: ptr::NonNull<GreenNodeAllocation>,
    child_count: usize,
    initialized_children: usize,
) -> ptr::NonNull<GreenNodeAllocation> {
    let compact_layout = allocation_layout(child_count, false);
    let wide_layout = allocation_layout(child_count, true);
    let buffer = unsafe { allocator::allocate(wide_layout) };
    let wide_allocation = buffer.cast::<GreenNodeAllocation>();
    unsafe {
        ptr::write(ptr::addr_of_mut!((*wide_allocation.as_ptr()).count), GreenNodeRefCount::new(1))
    };

    let old_parent =
        unsafe { ptr::NonNull::new_unchecked(ptr::addr_of_mut!((*allocation.as_ptr()).data)) };
    let old_children = unsafe { allocation_packed_child_ptr(allocation) };
    let new_children = unsafe { allocation_child_ptr(wide_allocation, child_count, true) };
    for index in 0..initialized_children {
        let child = unsafe { old_children.add(index).read().into_element(old_parent) };
        unsafe { ptr::write(new_children.add(index), GreenChild::from_element(child)) };
    }

    let checkpoints = initialized_children.saturating_sub(1) / CHILDREN_PER_CHECKPOINT;
    let old_checkpoints = unsafe { old_children.add(child_count).cast::<TextSize>() };
    let new_checkpoints = unsafe { new_children.add(child_count).cast::<TextSize>() };
    unsafe { ptr::copy_nonoverlapping(old_checkpoints, new_checkpoints, checkpoints) };
    unsafe { allocator::deallocate(allocation.cast(), compact_layout) };
    wide_allocation
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
        let old_size = unsafe { allocation.as_ref() }.count.fetch_sub(1, Release);
        if old_size != 1 {
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

        #[cfg(target_pointer_width = "64")]
        let wide = self.is_wide();
        #[cfg(not(target_pointer_width = "64"))]
        let wide = self.header.child_count == WIDE_CHILD_COUNT;
        let child_count = self.child_count();
        #[cfg(target_pointer_width = "64")]
        if !wide {
            let parent = self.ptr;
            for index in 0..child_count {
                // SAFETY: This is the final strong reference, so every packed child can be
                // reconstructed and dropped exactly once.
                unsafe { (*self.packed_child_ptr().add(index)).drop_owned(parent) };
            }
        } else {
            for index in 0..child_count {
                // SAFETY: Wide nodes store initialized owning child pointers.
                unsafe {
                    ptr::drop_in_place(self.wide_child_ptr().add(index).cast_mut());
                }
            }
        }
        #[cfg(not(target_pointer_width = "64"))]
        for index in 0..child_count {
            // SAFETY: This is the final strong reference.
            unsafe {
                ptr::drop_in_place(self.wide_child_ptr().add(index).cast_mut());
            }
        }
        // SAFETY: The count marker was initialized during construction and is dropped once.
        #[cfg(target_pointer_width = "64")]
        unsafe {
            ptr::drop_in_place(ptr::addr_of_mut!((*self.ptr.as_ptr())._c))
        };
        // SAFETY: The header was initialized during construction and is dropped once.
        #[cfg(not(target_pointer_width = "64"))]
        unsafe {
            ptr::drop_in_place(ptr::addr_of_mut!((*self.ptr.as_ptr()).header))
        };
        // SAFETY: `allocation` was allocated with this exact layout and all fields are dropped.
        unsafe { allocator::deallocate(allocation.cast(), allocation_layout(child_count, wide)) };
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
        self.kind().hash(state);
        self.child_count().hash(state);
        self.text_len().hash(state);
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
        #[cfg(target_pointer_width = "64")]
        let mut wide = u32::from(kind.0) >= PACKED_KIND_LIMIT
            || stored_child_count >= PACKED_CHILD_COUNT_LIMIT;
        #[cfg(not(target_pointer_width = "64"))]
        let wide = child_count >= WIDE_CHILD_COUNT as usize;
        #[cfg(not(target_pointer_width = "64"))]
        let inline_child_count = if child_count >= WIDE_CHILD_COUNT as usize {
            WIDE_CHILD_COUNT
        } else {
            child_count as u16
        };
        let layout = allocation_layout(child_count, wide);
        // SAFETY: `layout` is non-zero and valid.
        let buffer = unsafe { allocator::allocate(layout) };
        let mut allocation = buffer.cast::<GreenNodeAllocation>();
        let mut guard =
            GreenNodeAllocGuard { allocation, child_count, wide, initialized_children: 0 };
        // SAFETY: The allocation is valid and properly aligned for this write.
        unsafe {
            ptr::write(ptr::addr_of_mut!((*allocation.as_ptr()).count), GreenNodeRefCount::new(1))
        };

        let mut text_len: TextSize = 0.into();
        #[cfg(not(target_pointer_width = "64"))]
        if inline_child_count == WIDE_CHILD_COUNT {
            // SAFETY: Wide layouts reserve one aligned slot before the child tail.
            unsafe {
                ptr::write(
                    ptr::addr_of_mut!((*allocation.as_ptr()).data.children).cast::<u32>(),
                    stored_child_count,
                )
            };
        }
        for index in 0..child_count {
            let element = children.next().expect("ExactSizeIterator over-reported length");
            let element_len = element.text_len();

            #[cfg(target_pointer_width = "64")]
            if wide {
                let child_ptr = unsafe { allocation_child_ptr(allocation, child_count, true) };
                if index != 0 && index % CHILDREN_PER_CHECKPOINT == 0 {
                    let checkpoint_ptr = unsafe { child_ptr.add(child_count).cast::<TextSize>() };
                    unsafe {
                        ptr::write(
                            checkpoint_ptr.add(index / CHILDREN_PER_CHECKPOINT - 1),
                            text_len,
                        )
                    };
                }
                unsafe { ptr::write(child_ptr.add(index), GreenChild::from_element(element)) };
            } else {
                let parent = unsafe {
                    ptr::NonNull::new_unchecked(ptr::addr_of_mut!((*allocation.as_ptr()).data))
                };
                match PackedGreenChild::try_from_element(parent, element) {
                    Ok(child) => {
                        let child_ptr = unsafe { allocation_packed_child_ptr(allocation) };
                        if index != 0 && index % CHILDREN_PER_CHECKPOINT == 0 {
                            let checkpoint_ptr =
                                unsafe { child_ptr.add(child_count).cast::<TextSize>() };
                            unsafe {
                                ptr::write(
                                    checkpoint_ptr.add(index / CHILDREN_PER_CHECKPOINT - 1),
                                    text_len,
                                )
                            };
                        }
                        unsafe { ptr::write(child_ptr.add(index), child) };
                    }
                    Err(element) => {
                        allocation = unsafe { promote_to_wide(allocation, child_count, index) };
                        guard.allocation = allocation;
                        guard.wide = true;
                        wide = true;

                        let child_ptr =
                            unsafe { allocation_child_ptr(allocation, child_count, true) };
                        if index != 0 && index % CHILDREN_PER_CHECKPOINT == 0 {
                            let checkpoint_ptr =
                                unsafe { child_ptr.add(child_count).cast::<TextSize>() };
                            unsafe {
                                ptr::write(
                                    checkpoint_ptr.add(index / CHILDREN_PER_CHECKPOINT - 1),
                                    text_len,
                                )
                            };
                        }
                        unsafe {
                            ptr::write(child_ptr.add(index), GreenChild::from_element(element))
                        };
                    }
                }
            }
            #[cfg(not(target_pointer_width = "64"))]
            {
                let child_ptr = unsafe { allocation_child_ptr(allocation, child_count, wide) };
                if index != 0 && index % CHILDREN_PER_CHECKPOINT == 0 {
                    let checkpoint_ptr = unsafe { child_ptr.add(child_count).cast::<TextSize>() };
                    unsafe {
                        ptr::write(
                            checkpoint_ptr.add(index / CHILDREN_PER_CHECKPOINT - 1),
                            text_len,
                        )
                    };
                }
                unsafe { ptr::write(child_ptr.add(index), GreenChild::from_element(element)) };
            }

            text_len += element_len;
            guard.initialized_children += 1;
        }
        assert!(children.next().is_none(), "ExactSizeIterator under-reported length");

        #[cfg(target_pointer_width = "64")]
        if !wide && u32::from(text_len) >= PACKED_TEXT_LEN_LIMIT {
            allocation = unsafe { promote_to_wide(allocation, child_count, child_count) };
            wide = true;
            guard.allocation = allocation;
            guard.wide = true;
        }

        // SAFETY: Header initialization completes the allocation before it is published.
        #[cfg(target_pointer_width = "64")]
        unsafe {
            let data = ptr::addr_of_mut!((*allocation.as_ptr()).data);
            if wide {
                ptr::write(
                    ptr::addr_of_mut!((*data).children).cast::<WideGreenNodeHead>(),
                    WideGreenNodeHead {
                        child_count: stored_child_count,
                        text_len,
                        kind,
                        _padding: [0; 6],
                    },
                );
                ptr::write(ptr::addr_of_mut!((*data).packed_head), WIDE_PACKED_HEAD);
            } else {
                let packed_head = u32::from(kind.0)
                    | stored_child_count << PACKED_KIND_BITS
                    | u32::from(text_len) << (PACKED_KIND_BITS + PACKED_CHILD_COUNT_BITS);
                ptr::write(ptr::addr_of_mut!((*data).packed_head), packed_head);
            }
            ptr::write(ptr::addr_of_mut!((*data)._c), Count::new());
        }
        // SAFETY: Header initialization completes the allocation before it is published.
        #[cfg(not(target_pointer_width = "64"))]
        unsafe {
            ptr::write(
                ptr::addr_of_mut!((*allocation.as_ptr()).data.header),
                GreenNodeHead { kind, child_count: inline_child_count, text_len, _c: Count::new() },
            )
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

#[cfg(target_pointer_width = "64")]
impl PackedGreenChild {
    const TOKEN_TAG: u32 = 1;
    const MIN_OFFSET_UNITS: isize = -(1 << 30);
    const MAX_OFFSET_UNITS: isize = (1 << 30) - 1;

    fn try_from_element(
        parent: ptr::NonNull<GreenNodeData>,
        element: GreenElement,
    ) -> Result<Self, GreenElement> {
        let (child_addr, is_token) = match &element {
            NodeOrToken::Node(node) => (node.ptr.as_ptr().expose_provenance(), false),
            NodeOrToken::Token(token) => {
                ((&**token as *const GreenTokenData).expose_provenance(), true)
            }
        };
        let parent_addr = parent.addr().get();
        let offset = child_addr.wrapping_sub(parent_addr) as isize;
        debug_assert_eq!(offset & 3, 0);
        let offset_units = offset / 4;
        if !(Self::MIN_OFFSET_UNITS..=Self::MAX_OFFSET_UNITS).contains(&offset_units) {
            return Err(element);
        }

        let tag = if is_token { Self::TOKEN_TAG } else { 0 };
        let packed = Self((offset_units as i32 as u32) << 1 | tag);
        match element {
            NodeOrToken::Node(node) => {
                _ = GreenNode::into_raw(node);
            }
            NodeOrToken::Token(token) => {
                _ = GreenToken::into_raw(token);
            }
        }
        Ok(packed)
    }

    #[inline]
    fn is_token(self) -> bool {
        self.0 & Self::TOKEN_TAG != 0
    }

    #[inline]
    fn untagged<T>(self, parent: ptr::NonNull<GreenNodeData>) -> ptr::NonNull<T> {
        let offset = ((self.0 as i32) >> 1) as isize * 4;
        let addr = parent.addr().get().wrapping_add_signed(offset);
        // SAFETY: Construction exposed the owned child pointer's provenance and stored its
        // complete signed offset from the parent.
        unsafe { ptr::NonNull::new_unchecked(ptr::with_exposed_provenance_mut(addr)) }
    }

    #[inline]
    unsafe fn as_ref<'a>(self, parent: ptr::NonNull<GreenNodeData>) -> GreenElementRef<'a> {
        if self.is_token() {
            let ptr = self.untagged::<GreenTokenData>(parent);
            // SAFETY: The packed child owns this token reference.
            NodeOrToken::Token(unsafe { ptr.as_ref() })
        } else {
            let ptr = self.untagged::<GreenNodeData>(parent);
            // SAFETY: The packed child owns this node reference.
            NodeOrToken::Node(unsafe { ptr.as_ref() })
        }
    }

    fn into_element(self, parent: ptr::NonNull<GreenNodeData>) -> GreenElement {
        if self.is_token() {
            // SAFETY: Construction transferred exactly one owned token reference.
            NodeOrToken::Token(unsafe { GreenToken::from_raw(self.untagged(parent)) })
        } else {
            // SAFETY: Construction transferred exactly one owned node reference.
            NodeOrToken::Node(unsafe { GreenNode::from_raw(self.untagged(parent)) })
        }
    }

    fn drop_owned(self, parent: ptr::NonNull<GreenNodeData>) {
        drop(self.into_element(parent));
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
    node: &'a GreenNodeData,
    checkpoints: &'a [TextSize],
    front: usize,
    back: usize,
    front_offset: TextSize,
    back_offset: TextSize,
}

impl<'a> GreenChildren<'a> {
    #[inline]
    fn new(node: &'a GreenNodeData) -> Self {
        let child_count = node.child_count();
        GreenChildren {
            node,
            checkpoints: node.checkpoints_with_count(child_count),
            front: 0,
            back: child_count,
            front_offset: 0.into(),
            back_offset: node.text_len(),
        }
    }

    #[inline]
    fn child_at(&self, index: usize) -> GreenChildRef<'a> {
        let block = index / CHILDREN_PER_CHECKPOINT;
        let block_start = block * CHILDREN_PER_CHECKPOINT;
        let mut rel_offset = if block == 0 { 0.into() } else { self.checkpoints[block - 1] };
        for child_index in block_start..index {
            rel_offset += self.node.child_ref(child_index).text_len();
        }
        GreenChildRef { element: self.node.child_ref(index), rel_offset }
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
        let element = self.node.child_ref(self.front);
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
        let element = self.node.child_ref(self.back);
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
