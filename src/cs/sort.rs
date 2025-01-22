pub mod bubblesort;
pub mod bucketsort;
pub mod countingsort;
pub mod heapsort;
pub mod insertionsort;
pub mod mergesort;
pub mod quicksort;
pub mod radixsort;
pub mod selectionsort;
pub mod shellsort;

// Re-export sorting algorithms with descriptive names
pub use bubblesort::sort as bubble_sort;
pub use bucketsort::sort as bucket_sort;
pub use countingsort::sort as counting_sort;
#[cfg(feature = "simd")]
pub use heapsort::sort_i32 as heap_sort_i32;
pub use heapsort::{sort as heap_sort, HeapSortError};
pub use insertionsort::sort as insertion_sort;
pub use mergesort::{sort as merge_sort, MergeSortBuilder};
pub use quicksort::sort as quick_sort;
pub use radixsort::sort as radix_sort;
pub use selectionsort::sort as selection_sort;
pub use shellsort::sort as shell_sort;
