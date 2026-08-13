use std::{
    alloc::{self, Layout},
    collections::HashMap,
    ptr::NonNull,
    sync::{Mutex, OnceLock},
};

pub(super) const PAGE_SIZE: usize = 64 * 1024;
const ALIGN: usize = 4;
const MAX_POOLED_SIZE: usize = 256;
const CLASS_COUNT: usize = MAX_POOLED_SIZE / ALIGN;
const NONE: u32 = u32::MAX;

#[derive(Debug)]
struct Page {
    base: usize,
    free_head: u32,
    live: u32,
    capacity: u32,
    available_index: Option<usize>,
}

#[derive(Debug, Default)]
struct Pool {
    pages: Vec<Page>,
    pages_by_base: HashMap<usize, usize>,
    available: Vec<usize>,
}

static POOLS: OnceLock<Vec<Mutex<Pool>>> = OnceLock::new();

fn pools() -> &'static [Mutex<Pool>] {
    POOLS.get_or_init(|| (0..CLASS_COUNT).map(|_| Mutex::new(Pool::default())).collect())
}

fn class(layout: Layout) -> Option<(usize, usize)> {
    if layout.align() > ALIGN || layout.size() > MAX_POOLED_SIZE {
        return None;
    }
    let block_size = layout.size().max(ALIGN).next_multiple_of(ALIGN);
    Some((block_size / ALIGN - 1, block_size))
}

fn page_layout() -> Layout {
    Layout::from_size_align(PAGE_SIZE, PAGE_SIZE).unwrap()
}

impl Pool {
    unsafe fn add_page(&mut self, block_size: usize) {
        let layout = page_layout();
        // SAFETY: `layout` has non-zero size and valid alignment.
        let base = unsafe { alloc::alloc(layout) };
        let Some(base) = NonNull::new(base) else { alloc::handle_alloc_error(layout) };
        let capacity = PAGE_SIZE / block_size;
        for index in 0..capacity {
            let next = if index + 1 == capacity { NONE } else { (index + 1) as u32 };
            // SAFETY: Every block is at least four bytes, lies within the new page, and is not yet
            // exposed to a caller.
            unsafe { base.as_ptr().add(index * block_size).cast::<u32>().write(next) };
        }

        let base = base.addr().get();
        let page_index = self.pages.len();
        let available_index = self.available.len();
        self.pages.push(Page {
            base,
            free_head: 0,
            live: 0,
            capacity: capacity as u32,
            available_index: Some(available_index),
        });
        self.pages_by_base.insert(base, page_index);
        self.available.push(base);
    }

    unsafe fn allocate(&mut self, block_size: usize) -> NonNull<u8> {
        if self.available.is_empty() {
            // SAFETY: The caller selected a valid pooled block size.
            unsafe { self.add_page(block_size) };
        }
        let base = *self.available.last().unwrap();
        let page_index = self.pages_by_base[&base];
        let page = &mut self.pages[page_index];
        let block_index = page.free_head;
        debug_assert_ne!(block_index, NONE);
        let block = (base + block_index as usize * block_size) as *mut u8;
        // SAFETY: A free block stores its initialized next index in its first four bytes.
        page.free_head = unsafe { block.cast::<u32>().read() };
        page.live += 1;
        if page.free_head == NONE {
            let popped = self.available.pop();
            debug_assert_eq!(popped, Some(base));
            page.available_index = None;
        }
        // SAFETY: Page allocation and block arithmetic guarantee a non-null aligned pointer.
        unsafe { NonNull::new_unchecked(block) }
    }

    unsafe fn deallocate(&mut self, ptr: NonNull<u8>, block_size: usize) {
        let base = ptr.addr().get() & !(PAGE_SIZE - 1);
        let page_index = self.pages_by_base[&base];
        let page = &mut self.pages[page_index];
        let offset = ptr.addr().get() - base;
        debug_assert_eq!(offset % block_size, 0);
        let block_index = offset / block_size;
        debug_assert!(block_index < page.capacity as usize);
        let was_full = page.free_head == NONE;
        // SAFETY: `ptr` names a live block returned by this pool. Its caller has ended all typed
        // access, so the first four bytes can hold the free-list link again.
        unsafe { ptr.as_ptr().cast::<u32>().write(page.free_head) };
        page.free_head = block_index as u32;
        page.live -= 1;

        if was_full {
            page.available_index = Some(self.available.len());
            self.available.push(base);
        }
        if page.live == 0 && self.pages.len() > 1 {
            self.remove_page(page_index);
        }
    }

    fn remove_page(&mut self, page_index: usize) {
        let page = &self.pages[page_index];
        let available_index = page.available_index.expect("empty page must be available");
        let base = page.base;
        self.available.swap_remove(available_index);
        if let Some(&moved_base) = self.available.get(available_index) {
            let moved_page = self.pages_by_base[&moved_base];
            self.pages[moved_page].available_index = Some(available_index);
        }

        self.pages_by_base.remove(&base);
        self.pages.swap_remove(page_index);
        if let Some(moved) = self.pages.get(page_index) {
            self.pages_by_base.insert(moved.base, page_index);
        }
        // SAFETY: This empty page was allocated with `page_layout`, has been removed from every
        // pool index, and no live block points into it.
        unsafe { alloc::dealloc(base as *mut u8, page_layout()) };
    }
}

/// Allocates uninitialized storage for `layout`.
///
/// # Safety
///
/// The caller must initialize bytes before typed reads and later pass the returned pointer and the
/// exact same layout to [`deallocate`].
pub(super) unsafe fn allocate(layout: Layout) -> NonNull<u8> {
    let Some((class, block_size)) = class(layout) else {
        // SAFETY: `layout` has non-zero size and valid alignment.
        let ptr = unsafe { alloc::alloc(layout) };
        return NonNull::new(ptr).unwrap_or_else(|| alloc::handle_alloc_error(layout));
    };
    let mut pool = pools()[class].lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    // SAFETY: `class` and `block_size` were derived from the supplied layout.
    unsafe { pool.allocate(block_size) }
}

/// Releases storage returned by [`allocate`].
///
/// # Safety
///
/// `ptr` must be live, must have been returned by [`allocate`] for the exact same layout, and no
/// typed or aliased access may remain.
pub(super) unsafe fn deallocate(ptr: NonNull<u8>, layout: Layout) {
    let Some((class, block_size)) = class(layout) else {
        // SAFETY: The caller guarantees the exact allocation layout and ended all access.
        unsafe { alloc::dealloc(ptr.as_ptr(), layout) };
        return;
    };
    let mut pool = pools()[class].lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    // SAFETY: The caller guarantees this live block belongs to the selected class.
    unsafe { pool.deallocate(ptr, block_size) };
}
