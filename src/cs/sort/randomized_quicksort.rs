use rand::Rng;

pub fn randomized_quicksort<T: Ord + Clone>(arr: &mut [T]) {
    if arr.len() <= 1 {
        return;
    }
    let pivot_index = rand::thread_rng().gen_range(0..arr.len());
    arr.swap(pivot_index, arr.len() - 1);
    let pivot = arr[arr.len() - 1].clone();
    let mut i = 0;
    for j in 0..arr.len()-1 {
        if arr[j] <= pivot {
            arr.swap(i, j);
            i += 1;
        }
    }
    arr.swap(i, arr.len()-1);
    randomized_quicksort(&mut arr[0..i]);
    randomized_quicksort(&mut arr[i+1..]);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_randomized_quicksort() {
        let mut arr = vec![3, 6, 2, 7, 1, 8, 5, 4];
        randomized_quicksort(&mut arr);
        assert_eq!(arr, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }
}
