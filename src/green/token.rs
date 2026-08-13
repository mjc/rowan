use std::{
    alloc::Layout,
    borrow::Borrow,
    fmt,
    hash::{Hash, Hasher},
    mem::{self, ManuallyDrop},
    ops, ptr, slice,
    sync::atomic::{
        AtomicU32,
        Ordering::{Acquire, Relaxed, Release},
    },
};

use countme::Count;
use memoffset::offset_of;

use crate::{
    green::{allocator, SyntaxKind},
    TextSize,
};

const MAX_REFCOUNT: u32 = i32::MAX as u32;
const WIDE_TEXT_LEN: u16 = u16::MAX;

#[repr(C)]
struct GreenTokenAllocation {
    count: AtomicU32,
    data: GreenTokenData,
}

#[repr(C)]
pub struct GreenTokenData {
    /// The low half stores the syntax kind and the high half stores the common text length.
    /// `u16::MAX` in the high half indicates a full `u32` length before the trailing text.
    packed_head: u32,
    _c: Count<GreenToken>,
    text: [u8; 0],
}

impl PartialEq for GreenTokenData {
    fn eq(&self, other: &Self) -> bool {
        self.kind() == other.kind() && self.text() == other.text()
    }
}

impl Eq for GreenTokenData {}

impl Hash for GreenTokenData {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.kind().hash(state);
        self.text().hash(state);
    }
}

/// Leaf node in the immutable tree.
#[repr(transparent)]
pub struct GreenToken {
    ptr: ptr::NonNull<GreenTokenData>,
}

impl ToOwned for GreenTokenData {
    type Owned = GreenToken;

    #[inline]
    fn to_owned(&self) -> GreenToken {
        // SAFETY: `self` points into a live token allocation.
        let green = unsafe { GreenToken::from_raw(ptr::NonNull::from(self)) };
        let green = ManuallyDrop::new(green);
        GreenToken::clone(&green)
    }
}

impl Borrow<GreenTokenData> for GreenToken {
    #[inline]
    fn borrow(&self) -> &GreenTokenData {
        self
    }
}

impl fmt::Debug for GreenTokenData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GreenToken")
            .field("kind", &self.kind())
            .field("text", &self.text())
            .finish()
    }
}

impl fmt::Debug for GreenToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl fmt::Display for GreenToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

impl fmt::Display for GreenTokenData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text())
    }
}

impl GreenTokenData {
    #[inline]
    fn stored_text_len(&self) -> u16 {
        (self.packed_head >> u16::BITS) as u16
    }

    #[inline]
    fn is_wide(&self) -> bool {
        self.stored_text_len() == WIDE_TEXT_LEN
    }

    /// Kind of this Token.
    #[inline]
    pub fn kind(&self) -> SyntaxKind {
        SyntaxKind(self.packed_head as u16)
    }

    /// Text of this Token.
    #[inline]
    pub fn text(&self) -> &str {
        let stored_text_len = self.stored_text_len();
        let (ptr, len) = if stored_text_len == WIDE_TEXT_LEN {
            // SAFETY: Wide construction writes a `u32` at this aligned location.
            let len = unsafe { self.text.as_ptr().cast::<u32>().read() };
            (self.text.as_ptr().wrapping_add(mem::size_of::<u32>()), len)
        } else {
            (self.text.as_ptr(), u32::from(stored_text_len))
        };
        let len = usize::try_from(len).unwrap();
        // SAFETY: Construction copies exactly `len` bytes from a valid UTF-8 `str`.
        unsafe { std::str::from_utf8_unchecked(slice::from_raw_parts(ptr, len)) }
    }

    /// Returns the length of the text covered by this token.
    #[inline]
    pub fn text_len(&self) -> TextSize {
        if self.is_wide() {
            // SAFETY: Wide construction writes a `u32` at this aligned location.
            TextSize::new(unsafe { self.text.as_ptr().cast::<u32>().read() })
        } else {
            TextSize::new(u32::from(self.stored_text_len()))
        }
    }
}

pub(super) fn allocation_layout(text_len: usize, wide: bool) -> Layout {
    let text_offset = offset_of!(GreenTokenAllocation, data) + offset_of!(GreenTokenData, text);
    let usable_size = text_offset
        .checked_add(mem::size_of::<u32>() * usize::from(wide))
        .and_then(|size| size.checked_add(text_len))
        .expect("green token allocation size overflows");
    let align = mem::align_of::<GreenTokenAllocation>();
    let size = usable_size.checked_add(align - 1).unwrap() & !(align - 1);
    Layout::from_size_align(size, align).expect("invalid green token allocation layout")
}

#[inline]
unsafe fn allocation_ptr(data: ptr::NonNull<GreenTokenData>) -> ptr::NonNull<GreenTokenAllocation> {
    unsafe {
        data.cast::<u8>().sub(offset_of!(GreenTokenAllocation, data)).cast::<GreenTokenAllocation>()
    }
}

impl GreenToken {
    /// Creates new Token.
    #[inline]
    pub fn new(kind: SyntaxKind, text: &str) -> GreenToken {
        let text_len = text.len();
        let full_text_len = u32::try_from(text_len).expect("green token text exceeds u32::MAX");
        let wide = text_len >= usize::from(WIDE_TEXT_LEN);
        let stored_text_len = if wide { WIDE_TEXT_LEN } else { text_len as u16 };
        let packed_head = u32::from(kind.0) | (u32::from(stored_text_len) << u16::BITS);
        let layout = allocation_layout(text_len, wide);
        // SAFETY: `layout` has non-zero size and valid alignment.
        let buffer = unsafe { allocator::allocate(layout) };
        let allocation = buffer.cast::<GreenTokenAllocation>();
        // SAFETY: The allocation reserves each field and the complete trailing text.
        unsafe {
            ptr::write(ptr::addr_of_mut!((*allocation.as_ptr()).count), AtomicU32::new(1));
            let data = ptr::addr_of_mut!((*allocation.as_ptr()).data);
            ptr::write(ptr::addr_of_mut!((*data).packed_head), packed_head);
            ptr::write(ptr::addr_of_mut!((*data)._c), Count::new());
            let mut text_ptr = ptr::addr_of_mut!((*data).text).cast::<u8>();
            if wide {
                text_ptr.cast::<u32>().write(full_text_len);
                text_ptr = text_ptr.add(mem::size_of::<u32>());
            }
            ptr::copy_nonoverlapping(text.as_ptr(), text_ptr, text_len);
        }

        let ptr =
            unsafe { ptr::NonNull::new_unchecked(ptr::addr_of_mut!((*allocation.as_ptr()).data)) };
        GreenToken { ptr }
    }

    #[inline]
    pub(crate) fn into_raw(this: GreenToken) -> ptr::NonNull<GreenTokenData> {
        ManuallyDrop::new(this).ptr
    }

    #[inline]
    pub(crate) unsafe fn from_raw(ptr: ptr::NonNull<GreenTokenData>) -> GreenToken {
        GreenToken { ptr }
    }
}

impl Clone for GreenToken {
    #[inline]
    fn clone(&self) -> Self {
        // SAFETY: `self` keeps the allocation alive.
        let allocation = unsafe { allocation_ptr(self.ptr).as_ref() };
        let old_size = allocation.count.fetch_add(1, Relaxed);
        if old_size > MAX_REFCOUNT {
            std::process::abort();
        }
        GreenToken { ptr: self.ptr }
    }
}

impl Drop for GreenToken {
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

impl GreenToken {
    #[inline(never)]
    unsafe fn drop_slow(&mut self, allocation: ptr::NonNull<GreenTokenAllocation>) {
        unsafe { allocation.as_ref() }.count.load(Acquire);
        let text_len = usize::try_from(u32::from(self.text_len())).unwrap();
        let wide = self.is_wide();
        // SAFETY: The count marker was initialized during construction and is dropped once.
        unsafe { ptr::drop_in_place(ptr::addr_of_mut!((*self.ptr.as_ptr())._c)) };
        // SAFETY: `allocation` was allocated with this exact layout and all fields are dropped.
        unsafe { allocator::deallocate(allocation.cast(), allocation_layout(text_len, wide)) };
    }
}

impl PartialEq for GreenToken {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(&**self, &**other) || **self == **other
    }
}

impl Eq for GreenToken {}

impl Hash for GreenToken {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

impl ops::Deref for GreenToken {
    type Target = GreenTokenData;

    #[inline]
    fn deref(&self) -> &GreenTokenData {
        // SAFETY: `self` owns a strong reference to this immutable allocation.
        unsafe { self.ptr.as_ref() }
    }
}

// SAFETY: Green tokens are immutable and their reference count is atomic.
unsafe impl Send for GreenToken {}
// SAFETY: Green tokens are immutable and their reference count is atomic.
unsafe impl Sync for GreenToken {}
