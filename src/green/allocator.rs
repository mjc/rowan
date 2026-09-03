use std::{
    alloc::{self, Layout},
    collections::HashMap,
    mem::MaybeUninit,
    ptr::NonNull,
    sync::{Mutex, OnceLock},
};

pub(super) const PAGE_SIZE: usize = 64 * 1024;
const CHUNK_SIZE: usize = 4096;
const CHUNK_COUNT: usize = PAGE_SIZE / CHUNK_SIZE;
const ALIGN: usize = 4;
const MAX_POOLED_SIZE: usize = 256;
const CLASS_COUNT: usize = MAX_POOLED_SIZE / ALIGN;
const NONE: u16 = u16::MAX;
const ALL_CHUNKS: u16 = u16::MAX;

const _: () = assert!(PAGE_SIZE.is_power_of_two());
const _: () = assert!(CHUNK_COUNT == u16::BITS as usize);
const _: () = assert!(CHUNK_COUNT * (CHUNK_SIZE / ALIGN) < NONE as usize);

#[repr(C, align(65536))]
struct AlignedPage {
    _bytes: [MaybeUninit<u8>; PAGE_SIZE],
}

struct Page {
    storage: Box<MaybeUninit<AlignedPage>>,
    base: usize,
    free_heads: [u16; CHUNK_COUNT],
    live_by_chunk: [u16; CHUNK_COUNT],
    reclaimed_chunks: u16,
    live: u32,
    available_index: Option<usize>,
}

#[derive(Default)]
struct Pool {
    pages: Vec<Page>,
    pages_by_base: HashMap<usize, usize>,
    available: Vec<usize>,
}

static POOLS: OnceLock<Vec<Mutex<Pool>>> = OnceLock::new();

fn pools() -> &'static [Mutex<Pool>] {
    POOLS.get_or_init(|| (0..CLASS_COUNT).map(|_| Mutex::new(Pool::default())).collect())
}

#[cfg(feature = "pool-allocator")]
fn class(layout: Layout) -> Option<(usize, usize)> {
    if layout.align() > ALIGN || layout.size() > MAX_POOLED_SIZE {
        return None;
    }
    let block_size = layout.size().max(ALIGN).next_multiple_of(ALIGN);
    Some((block_size / ALIGN - 1, block_size))
}

#[cfg(not(feature = "pool-allocator"))]
fn class(_: Layout) -> Option<(usize, usize)> {
    None
}

fn blocks_per_chunk(block_size: usize) -> usize {
    CHUNK_SIZE / block_size
}

impl Page {
    fn has_available(&self) -> bool {
        self.reclaimed_chunks != 0 || self.free_heads.iter().any(|&head| head != NONE)
    }

    fn available_chunk(&self) -> usize {
        self.free_heads
            .iter()
            .position(|&head| head != NONE)
            .unwrap_or_else(|| self.reclaimed_chunks.trailing_zeros() as usize)
    }

    fn block_ptr(&mut self, block_index: usize, block_size: usize) -> *mut u8 {
        let blocks_per_chunk = blocks_per_chunk(block_size);
        let chunk = block_index / blocks_per_chunk;
        let block = block_index % blocks_per_chunk;
        (&raw mut *self.storage).cast::<u8>().wrapping_add(chunk * CHUNK_SIZE + block * block_size)
    }

    unsafe fn initialize_chunk(&mut self, chunk: usize, block_size: usize) {
        debug_assert_ne!(self.reclaimed_chunks & (1 << chunk), 0);
        let blocks_per_chunk = blocks_per_chunk(block_size);
        let first = chunk * blocks_per_chunk;
        for block in 0..blocks_per_chunk {
            let index = first + block;
            let next = if block + 1 == blocks_per_chunk { NONE } else { (index + 1) as u16 };
            // SAFETY: This reclaimed or fresh chunk has no live blocks. Every indexed block lies
            // wholly within it, is at least four bytes, and is not exposed to a caller.
            unsafe { self.block_ptr(index, block_size).cast::<u16>().write(next) };
        }
        self.free_heads[chunk] = first as u16;
        self.reclaimed_chunks &= !(1 << chunk);
    }

    fn block_index(&self, ptr: NonNull<u8>, block_size: usize) -> (usize, usize) {
        let offset = ptr.addr().get() - self.base;
        let chunk = offset / CHUNK_SIZE;
        let chunk_offset = offset % CHUNK_SIZE;
        debug_assert!(chunk < CHUNK_COUNT);
        debug_assert_eq!(chunk_offset % block_size, 0);
        let block = chunk_offset / block_size;
        let blocks_per_chunk = blocks_per_chunk(block_size);
        debug_assert!(block < blocks_per_chunk);
        (chunk, chunk * blocks_per_chunk + block)
    }
}

impl Pool {
    fn add_page(&mut self) {
        let storage = Box::<AlignedPage>::new_uninit();
        let base = (&raw const *storage).addr();
        let page_index = self.pages.len();
        let available_index = self.available.len();
        self.pages.push(Page {
            storage,
            base,
            free_heads: [NONE; CHUNK_COUNT],
            live_by_chunk: [0; CHUNK_COUNT],
            reclaimed_chunks: ALL_CHUNKS,
            live: 0,
            available_index: Some(available_index),
        });
        self.pages_by_base.insert(base, page_index);
        self.available.push(base);
    }

    unsafe fn allocate(&mut self, block_size: usize) -> NonNull<u8> {
        if self.available.is_empty() {
            self.add_page();
        }
        let base = *self.available.last().unwrap();
        let page_index = self.pages_by_base[&base];
        let page = &mut self.pages[page_index];
        let chunk = page.available_chunk();
        if page.reclaimed_chunks & (1 << chunk) != 0 {
            // SAFETY: Reclaimed chunks have no live allocations and are initialized before use.
            unsafe { page.initialize_chunk(chunk, block_size) };
        }
        let block_index = page.free_heads[chunk];
        debug_assert_ne!(block_index, NONE);
        let block = page.block_ptr(block_index as usize, block_size);
        // SAFETY: An initialized free block stores its next index in the first two bytes.
        page.free_heads[chunk] = unsafe { block.cast::<u16>().read() };
        page.live_by_chunk[chunk] += 1;
        page.live += 1;
        if !page.has_available() {
            let popped = self.available.pop();
            debug_assert_eq!(popped, Some(base));
            page.available_index = None;
        }
        // SAFETY: Page and chunk allocation guarantee a non-null, four-byte-aligned pointer.
        unsafe { NonNull::new_unchecked(block) }
    }

    unsafe fn deallocate(&mut self, ptr: NonNull<u8>, block_size: usize) {
        let base = ptr.addr().get() & !(PAGE_SIZE - 1);
        let page_index = self.pages_by_base[&base];
        let page = &mut self.pages[page_index];
        let (chunk, block_index) = page.block_index(ptr, block_size);
        debug_assert_ne!(page.reclaimed_chunks & (1 << chunk), 1 << chunk);
        let was_full = !page.has_available();
        // SAFETY: `ptr` names a live block whose caller ended all typed access, so its first two
        // bytes can hold the chunk-local free-list link.
        unsafe { ptr.as_ptr().cast::<u16>().write(page.free_heads[chunk]) };
        page.free_heads[chunk] = block_index as u16;
        page.live_by_chunk[chunk] -= 1;
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

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn madvise(address: *mut std::ffi::c_void, length: usize, advice: i32) -> i32;
}

#[cfg(target_os = "linux")]
const MADV_DONTNEED: i32 = 4;

/// Releases physical pages belonging to empty pool chunks while retaining their virtual address
/// ranges for later reuse.
pub(super) fn trim() -> usize {
    #[cfg(not(target_os = "linux"))]
    return 0;

    #[cfg(target_os = "linux")]
    {
        let Some(pools) = POOLS.get() else { return 0 };
        let mut reclaimed = 0;
        for pool in pools {
            let mut pool = pool.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            for page in &mut pool.pages {
                for chunk in 0..CHUNK_COUNT {
                    let chunk_bit = 1 << chunk;
                    if page.live_by_chunk[chunk] != 0 || page.reclaimed_chunks & chunk_bit != 0 {
                        continue;
                    }
                    let address = (&raw mut *page.storage)
                        .cast::<u8>()
                        .wrapping_add(chunk * CHUNK_SIZE)
                        .cast();
                    // SAFETY: Page bases and chunk offsets are 4 KiB aligned. The pool lock
                    // excludes allocation and deallocation in this size class, and the per-chunk
                    // live count proves no pointer names the discarded range. Free-list state is
                    // held in `Page` and will be rebuilt before this chunk is reused.
                    if unsafe { madvise(address, CHUNK_SIZE, MADV_DONTNEED) } == 0 {
                        page.free_heads[chunk] = NONE;
                        page.reclaimed_chunks |= chunk_bit;
                        reclaimed += CHUNK_SIZE;
                    }
                }
            }
        }
        reclaimed
    }
}
