use rand::Rng;

pub fn randomized_quickselect<T: Ord + Copy>(arr: &mut [T], k: usize) -> T {
    assert!(k < arr.len(), "k is out of bounds");
    if arr.len() == 1 {
        return arr[0];
    }
    let pivot_index = rand::thread_rng().gen_range(0..arr.len());
    arr.swap(pivot_index, arr.len() - 1);
    let pivot = arr[arr.len() - 1];
    let mut i = 0;
    for j in 0..arr.len()-1 {
        if arr[j] <= pivot {
            arr.swap(i, j);
            i += 1;
        }
    }
    arr.swap(i, arr.len()-1);
    if k == i {
        return arr[i];
    } else if k < i {
        randomized_quickselect(&mut arr[..i], k)
    } else {
        randomized_quickselect(&mut arr[i+1..], k - i - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_randomized_quickselect() {
        let mut arr = [7, 1, 3, 4, 6, 2, 5];
        let kth = randomized_quickselect(&mut arr, 3); // 0-indexed: 4th smallest element
        let mut sorted = arr;
        sorted.sort();
        assert_eq!(kth, sorted[3]);
    }
}
