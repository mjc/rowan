use std::{
    alloc::Layout,
    borrow::Borrow,
    collections::HashMap,
    fmt,
    hash::{Hash, Hasher},
    mem::{self, ManuallyDrop},
    ops, ptr, slice,
    sync::atomic::{
        fence, AtomicU32,
        Ordering::{Acquire, Relaxed, Release},
    },
    sync::{Mutex, OnceLock},
};

use countme::Count;
use memoffset::offset_of;

use crate::{
    green::{allocator, SyntaxKind},
    TextSize,
};

const MAX_REFCOUNT: u32 = i32::MAX as u32;
const FIELD_BITS: u32 = 10;
const FIELD_MASK: u32 = (1 << FIELD_BITS) - 1;
const REFCOUNT_SHIFT: u32 = FIELD_BITS * 2;
const REFCOUNT_ONE: u32 = 1 << REFCOUNT_SHIFT;
const REFCOUNT_OVERFLOW: u32 = (1 << (u32::BITS - REFCOUNT_SHIFT)) - 1;
const REFCOUNT_INLINE_MAX: u32 = REFCOUNT_OVERFLOW - 1;
const HOT_TOKEN_LEN: usize = 8;

static REFCOUNT_OVERFLOWS: OnceLock<Mutex<HashMap<usize, u32>>> = OnceLock::new();

fn refcount_overflows() -> &'static Mutex<HashMap<usize, u32>> {
    REFCOUNT_OVERFLOWS.get_or_init(|| Mutex::new(HashMap::new()))
}

#[repr(C)]
struct GreenTokenAllocation {
    data: GreenTokenData,
}

#[repr(C)]
struct WideGreenTokenHead {
    kind: u16,
    _padding: u16,
    text_len: u32,
}

#[repr(C)]
pub struct GreenTokenData {
    /// Stores the common syntax kind, text length, and reference count. An all-ones kind selects a
    /// full kind and text length before the trailing text.
    packed_head: AtomicU32,
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
    fn packed_head(&self) -> u32 {
        self.packed_head.load(Relaxed)
    }

    #[inline]
    fn is_wide(&self) -> bool {
        self.packed_head() & FIELD_MASK == FIELD_MASK
    }

    #[inline]
    unsafe fn wide_head(&self) -> &WideGreenTokenHead {
        // SAFETY: The caller checks the wide sentinel before reading the initialized wide header.
        unsafe { &*self.text.as_ptr().cast() }
    }

    #[inline]
    fn uses_hot_count(&self) -> bool {
        if self.is_wide() {
            // SAFETY: Wide construction initializes the header before exposing the token.
            unsafe { self.wide_head() }.text_len as usize <= HOT_TOKEN_LEN
        } else {
            ((self.packed_head() >> FIELD_BITS) & FIELD_MASK) as usize <= HOT_TOKEN_LEN
        }
    }

    #[inline]
    unsafe fn hot_count(&self) -> &AtomicU32 {
        let offset = mem::size_of::<WideGreenTokenHead>() * usize::from(self.is_wide());
        // SAFETY: Hot-token construction initializes this aligned count before the text bytes.
        unsafe { &*self.text.as_ptr().add(offset).cast() }
    }

    #[inline]
    fn text_ptr(&self) -> *const u8 {
        self.text
            .as_ptr()
            .wrapping_add(mem::size_of::<WideGreenTokenHead>() * usize::from(self.is_wide()))
            .wrapping_add(mem::size_of::<AtomicU32>() * usize::from(self.uses_hot_count()))
    }

    /// Kind of this Token.
    #[inline]
    pub fn kind(&self) -> SyntaxKind {
        let packed = self.packed_head();
        if packed & FIELD_MASK == FIELD_MASK {
            // SAFETY: Wide construction initializes the header before exposing the token.
            SyntaxKind(unsafe { self.wide_head() }.kind)
        } else {
            SyntaxKind((packed & FIELD_MASK) as u16)
        }
    }

    /// Text of this Token.
    #[inline]
    pub fn text(&self) -> &str {
        let packed = self.packed_head();
        let (ptr, len) = if packed & FIELD_MASK == FIELD_MASK {
            // SAFETY: Wide construction initializes the header before the text bytes.
            let wide = unsafe { self.wide_head() };
            (self.text_ptr(), wide.text_len)
        } else {
            (self.text_ptr(), (packed >> FIELD_BITS) & FIELD_MASK)
        };
        let len = usize::try_from(len).unwrap();
        // SAFETY: Construction copies exactly `len` bytes from a valid UTF-8 `str`.
        unsafe { std::str::from_utf8_unchecked(slice::from_raw_parts(ptr, len)) }
    }

    /// Returns the length of the text covered by this token.
    #[inline]
    pub fn text_len(&self) -> TextSize {
        if self.is_wide() {
            // SAFETY: Wide construction initializes the header before exposing the token.
            TextSize::new(unsafe { self.wide_head() }.text_len)
        } else {
            TextSize::new((self.packed_head() >> FIELD_BITS) & FIELD_MASK)
        }
    }
}

pub(super) fn allocation_layout(text_len: usize, wide: bool) -> Layout {
    let text_offset = offset_of!(GreenTokenAllocation, data) + offset_of!(GreenTokenData, text);
    let usable_size = text_offset
        .checked_add(mem::size_of::<WideGreenTokenHead>() * usize::from(wide))
        .and_then(|size| {
            size.checked_add(mem::size_of::<AtomicU32>() * usize::from(text_len <= HOT_TOKEN_LEN))
        })
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
        let wide = u32::from(kind.0) >= FIELD_MASK || text_len >= FIELD_MASK as usize;
        let hot = text_len <= HOT_TOKEN_LEN;
        let initial_count = if hot { REFCOUNT_OVERFLOW << REFCOUNT_SHIFT } else { REFCOUNT_ONE };
        let packed_head = if wide {
            FIELD_MASK | (FIELD_MASK << FIELD_BITS) | initial_count
        } else {
            u32::from(kind.0) | ((text_len as u32) << FIELD_BITS) | initial_count
        };
        let layout = allocation_layout(text_len, wide);
        // SAFETY: `layout` has non-zero size and valid alignment.
        let buffer = unsafe { allocator::allocate(layout) };
        let allocation = buffer.cast::<GreenTokenAllocation>();
        // SAFETY: The allocation reserves each field and the complete trailing text.
        unsafe {
            let data = ptr::addr_of_mut!((*allocation.as_ptr()).data);
            ptr::write(ptr::addr_of_mut!((*data).packed_head), AtomicU32::new(packed_head));
            ptr::write(ptr::addr_of_mut!((*data)._c), Count::new());
            let mut text_ptr = ptr::addr_of_mut!((*data).text).cast::<u8>();
            if wide {
                text_ptr.cast::<WideGreenTokenHead>().write(WideGreenTokenHead {
                    kind: kind.0,
                    _padding: 0,
                    text_len: full_text_len,
                });
                text_ptr = text_ptr.add(mem::size_of::<WideGreenTokenHead>());
            }
            if hot {
                text_ptr.cast::<AtomicU32>().write(AtomicU32::new(1));
                text_ptr = text_ptr.add(mem::size_of::<AtomicU32>());
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
        self.increment_refcount();
        GreenToken { ptr: self.ptr }
    }
}

impl Drop for GreenToken {
    #[inline]
    fn drop(&mut self) {
        if !self.decrement_refcount() {
            return;
        }
        // SAFETY: This was the final strong reference.
        unsafe { self.drop_slow() };
    }
}

impl GreenToken {
    fn increment_refcount(&self) {
        if self.uses_hot_count() {
            // SAFETY: Hot-token construction initializes this counter for the allocation lifetime.
            let old_size = unsafe { self.hot_count() }.fetch_add(1, Relaxed);
            if old_size > MAX_REFCOUNT {
                std::process::abort();
            }
            return;
        }
        loop {
            let head = self.packed_head.load(Relaxed);
            let count = head >> REFCOUNT_SHIFT;
            if count < REFCOUNT_INLINE_MAX {
                if self
                    .packed_head
                    .compare_exchange_weak(head, head + REFCOUNT_ONE, Relaxed, Relaxed)
                    .is_ok()
                {
                    return;
                }
                continue;
            }

            let mut overflows =
                refcount_overflows().lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            let head = self.packed_head.load(Relaxed);
            let count = head >> REFCOUNT_SHIFT;
            if count == REFCOUNT_INLINE_MAX {
                if self
                    .packed_head
                    .compare_exchange(head, head + REFCOUNT_ONE, Relaxed, Relaxed)
                    .is_ok()
                {
                    overflows.insert(self.ptr.as_ptr() as usize, REFCOUNT_OVERFLOW);
                    return;
                }
                continue;
            }
            if count != REFCOUNT_OVERFLOW {
                continue;
            }
            let count = overflows
                .get_mut(&(self.ptr.as_ptr() as usize))
                .expect("overflowed green token refcount must be tracked");
            if *count >= MAX_REFCOUNT {
                std::process::abort();
            }
            *count += 1;
            return;
        }
    }

    fn decrement_refcount(&self) -> bool {
        if self.uses_hot_count() {
            // SAFETY: Hot-token construction initializes this counter for the allocation lifetime.
            let old_size = unsafe { self.hot_count() }.fetch_sub(1, Release);
            if old_size != 1 {
                return false;
            }
            unsafe { self.hot_count() }.load(Acquire);
            return true;
        }
        loop {
            let head = self.packed_head.load(Relaxed);
            let count = head >> REFCOUNT_SHIFT;
            if count != REFCOUNT_OVERFLOW {
                debug_assert_ne!(count, 0);
                if self
                    .packed_head
                    .compare_exchange_weak(head, head - REFCOUNT_ONE, Release, Relaxed)
                    .is_ok()
                {
                    if count == 1 {
                        fence(Acquire);
                        return true;
                    }
                    return false;
                }
                continue;
            }

            let mut overflows =
                refcount_overflows().lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            if self.packed_head.load(Relaxed) >> REFCOUNT_SHIFT != REFCOUNT_OVERFLOW {
                continue;
            }
            let key = self.ptr.as_ptr() as usize;
            let count =
                overflows.get_mut(&key).expect("overflowed green token refcount must be tracked");
            if *count > REFCOUNT_OVERFLOW {
                *count -= 1;
                return false;
            }
            overflows.remove(&key);
            self.packed_head.fetch_sub(REFCOUNT_ONE, Release);
            return false;
        }
    }

    #[inline(never)]
    unsafe fn drop_slow(&mut self) {
        let text_len = usize::try_from(u32::from(self.text_len())).unwrap();
        let wide = self.is_wide();
        // SAFETY: The count marker was initialized during construction and is dropped once.
        unsafe { ptr::drop_in_place(ptr::addr_of_mut!((*self.ptr.as_ptr())._c)) };
        // SAFETY: The token was allocated with this exact layout and all fields are dropped.
        unsafe {
            allocator::deallocate(
                allocation_ptr(self.ptr).cast(),
                allocation_layout(text_len, wide),
            )
        };
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
