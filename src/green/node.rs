use std::{
    borrow::Borrow,
    fmt,
    hash::{Hash, Hasher},
    iter::{self, FusedIterator},
    mem::{self, ManuallyDrop},
    num::NonZeroUsize,
    ops, ptr, slice,
};

use countme::Count;

use crate::{
    arc::{Arc, HeaderSlice, ThinArc},
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

/// An owning node-or-token pointer and its offset.
///
/// The pointer's low bit distinguishes tokens from nodes. Four-byte packing keeps the offset from
/// adding pointer-alignment padding.
#[repr(C, packed(4))]
pub(crate) struct GreenChild {
    ptr: ptr::NonNull<()>,
    rel_offset: TextSize,
}
#[cfg(target_pointer_width = "64")]
static_assert!(mem::size_of::<GreenChild>() == 12);
static_assert!(mem::align_of::<GreenNodeData>() >= 2);
static_assert!(mem::align_of::<GreenTokenData>() >= 2);

type Repr = HeaderSlice<GreenNodeHead, [GreenChild]>;
type ReprThin = HeaderSlice<GreenNodeHead, [GreenChild; 0]>;
#[repr(transparent)]
pub struct GreenNodeData {
    data: ReprThin,
}

impl PartialEq for GreenNodeData {
    fn eq(&self, other: &Self) -> bool {
        self.header() == other.header() && self.slice() == other.slice()
    }
}

/// Internal node in the immutable tree.
/// It has other nodes and tokens as children.
#[derive(Clone, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct GreenNode {
    ptr: ThinArc<GreenNodeHead, GreenChild>,
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
        &self.data.header
    }

    #[inline]
    fn slice(&self) -> &[GreenChild] {
        self.data.slice()
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
        Children { raw: self.slice().iter() }
    }

    #[inline]
    pub(crate) fn child_at(&self, index: usize) -> Option<&GreenChild> {
        self.slice().get(index)
    }

    #[cfg(test)]
    pub(crate) fn child_offset(&self, index: usize) -> TextSize {
        self.slice()[index].rel_offset()
    }

    pub(crate) fn child_at_range(
        &self,
        rel_range: TextRange,
    ) -> Option<(usize, TextSize, GreenElementRef<'_>)> {
        let idx = self
            .slice()
            .binary_search_by(|it| {
                let child_range = it.rel_range();
                TextRange::ordering(child_range, rel_range)
            })
            // XXX: this handles empty ranges
            .unwrap_or_else(|it| it.saturating_sub(1));
        let child = &self.slice().get(idx).filter(|it| it.rel_range().contains_range(rel_range))?;
        Some((idx, child.rel_offset(), child.as_ref()))
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
        unsafe {
            let repr: &Repr = &self.ptr;
            let repr: &ReprThin = &*(repr as *const Repr as *const ReprThin);
            mem::transmute::<&ReprThin, &GreenNodeData>(repr)
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
        let mut text_len: TextSize = 0.into();
        let children = children.into_iter().map(|el| {
            let rel_offset = text_len;
            text_len += el.text_len();
            GreenChild::from_element(el, rel_offset)
        });

        let data = ThinArc::from_header_and_iter(
            GreenNodeHead { kind, text_len: 0.into(), _c: Count::new() },
            children,
        );

        // XXX: fixup `text_len` after construction, because we can't iterate
        // `children` twice.
        let data = {
            let mut data = Arc::from_thin(data);
            Arc::get_mut(&mut data).unwrap().header.text_len = text_len;
            Arc::into_thin(data)
        };

        GreenNode { ptr: data }
    }

    #[inline]
    pub(crate) fn into_raw(this: GreenNode) -> ptr::NonNull<GreenNodeData> {
        let green = ManuallyDrop::new(this);
        let green: &GreenNodeData = &*green;
        ptr::NonNull::from(&*green)
    }

    #[inline]
    pub(crate) unsafe fn from_raw(ptr: ptr::NonNull<GreenNodeData>) -> GreenNode {
        let arc = Arc::from_raw(&ptr.as_ref().data as *const ReprThin);
        let arc = mem::transmute::<Arc<ReprThin>, ThinArc<GreenNodeHead, GreenChild>>(arc);
        GreenNode { ptr: arc }
    }
}

impl GreenChild {
    const TOKEN_TAG: usize = 1;

    fn from_node(node: GreenNode, rel_offset: TextSize) -> GreenChild {
        let ptr = GreenNode::into_raw(node).cast();
        debug_assert_eq!(ptr.addr().get() & Self::TOKEN_TAG, 0);
        GreenChild { ptr, rel_offset }
    }

    fn from_token(token: GreenToken, rel_offset: TextSize) -> GreenChild {
        let ptr = GreenToken::into_raw(token).cast();
        debug_assert_eq!(ptr.addr().get() & Self::TOKEN_TAG, 0);
        let ptr = ptr.map_addr(|addr| NonZeroUsize::new(addr.get() | Self::TOKEN_TAG).unwrap());
        GreenChild { ptr, rel_offset }
    }

    fn from_element(element: GreenElement, rel_offset: TextSize) -> GreenChild {
        match element {
            NodeOrToken::Node(node) => GreenChild::from_node(node, rel_offset),
            NodeOrToken::Token(token) => GreenChild::from_token(token, rel_offset),
        }
    }

    fn ptr(&self) -> ptr::NonNull<()> {
        // SAFETY: `ptr` is initialized but packed to four-byte alignment.
        unsafe { ptr::addr_of!(self.ptr).read_unaligned() }
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

    #[inline]
    pub(crate) fn rel_offset(&self) -> TextSize {
        // SAFETY: `rel_offset` is initialized and naturally aligned by the four-byte packing.
        unsafe { ptr::addr_of!(self.rel_offset).read_unaligned() }
    }

    #[inline]
    fn rel_range(&self) -> TextRange {
        let len = self.as_ref().text_len();
        TextRange::at(self.rel_offset(), len)
    }
}

impl Clone for GreenChild {
    fn clone(&self) -> Self {
        GreenChild::from_element(self.as_ref().to_owned(), self.rel_offset())
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

impl fmt::Debug for GreenChild {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GreenChild")
            .field("rel_offset", &self.rel_offset())
            .field("element", &self.as_ref())
            .finish()
    }
}

impl PartialEq for GreenChild {
    fn eq(&self, other: &Self) -> bool {
        self.rel_offset() == other.rel_offset()
            && (self.ptr() == other.ptr() || self.as_ref() == other.as_ref())
    }
}

impl Eq for GreenChild {}

impl Hash for GreenChild {
    fn hash<H: Hasher>(&self, state: &mut H) {
        fn hash_element<H: Hasher>(
            element: GreenElementRef<'_>,
            rel_offset: TextSize,
            state: &mut H,
        ) {
            rel_offset.hash(state);
            match element {
                NodeOrToken::Node(node) => {
                    false.hash(state);
                    node.kind().hash(state);
                    node.text_len().hash(state);
                    for child in node.slice() {
                        hash_element(child.as_ref(), child.rel_offset(), state);
                    }
                }
                NodeOrToken::Token(token) => {
                    true.hash(state);
                    token.kind().hash(state);
                    token.text().hash(state);
                }
            }
        }

        hash_element(self.as_ref(), self.rel_offset(), state);
    }
}

// SAFETY: `GreenChild` owns an immutable `GreenNode` or `GreenToken`, both Send and Sync.
unsafe impl Send for GreenChild {}
// SAFETY: `GreenChild` owns an immutable `GreenNode` or `GreenToken`, both Send and Sync.
unsafe impl Sync for GreenChild {}

#[derive(Debug, Clone)]
pub struct Children<'a> {
    pub(crate) raw: slice::Iter<'a, GreenChild>,
}

// NB: forward everything stable that iter::Slice specializes as of Rust 1.39.0
impl ExactSizeIterator for Children<'_> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.raw.len()
    }
}

impl<'a> Iterator for Children<'a> {
    type Item = GreenElementRef<'a>;

    #[inline]
    fn next(&mut self) -> Option<GreenElementRef<'a>> {
        self.raw.next().map(GreenChild::as_ref)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.raw.size_hint()
    }

    #[inline]
    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.raw.count()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.raw.nth(n).map(GreenChild::as_ref)
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        self.next_back()
    }

    #[inline]
    fn fold<Acc, Fold>(mut self, init: Acc, mut f: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        let mut accum = init;
        while let Some(x) = self.next() {
            accum = f(accum, x);
        }
        accum
    }
}

impl<'a> DoubleEndedIterator for Children<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.raw.next_back().map(GreenChild::as_ref)
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.raw.nth_back(n).map(GreenChild::as_ref)
    }

    #[inline]
    fn rfold<Acc, Fold>(mut self, init: Acc, mut f: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        let mut accum = init;
        while let Some(x) = self.next_back() {
            accum = f(accum, x);
        }
        accum
    }
}

impl FusedIterator for Children<'_> {}
