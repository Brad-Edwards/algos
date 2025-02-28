use std::collections::HashMap;

/// Compresses the input data using the LZW algorithm.
///
/// # Parameters
///
/// - `input`: a slice of bytes to compress.
///
/// # Returns
///
/// A vector of u16 codes representing the compressed data.
///
/// # Details
///
/// The dictionary is initialized with all 256 possible single-byte sequences.
/// Then, the algorithm finds the longest sequence `w` present in the dictionary that
/// is a prefix of the remaining input. It outputs the code for `w`, adds `w` concatenated
/// with the next byte to the dictionary, and continues.
///
/// # Example
///
/// ```
/// use algos::cs::compression::lzw::compress;
///
/// let input = b"TOBEORNOTTOBE";
/// let compressed = compress(input);
/// assert!(!compressed.is_empty());
/// ```
pub fn compress(input: &[u8]) -> Vec<u16> {
    // Initialize the dictionary with all 256 single-byte sequences.
    let mut dict: HashMap<Vec<u8>, u16> = HashMap::new();
    for i in 0..256u16 {
        dict.insert(vec![i as u8], i);
    }
    let mut next_code = 256u16;

    let mut result = Vec::new();
    let mut w: Vec<u8> = Vec::new();
    let mut i = 0;
    while i < input.len() {
        // Extend w with the current byte.
        w.push(input[i]);
        // If w is not in the dictionary and w has length >= 1, then
        // the longest prefix is w without its last byte.
        if !dict.contains_key(&w) {
            // Remove last byte (the new addition)
            let mut prefix = w.clone();
            let last = prefix.pop().unwrap();
            // Output the code for the longest prefix.
            let code = dict.get(&prefix).copied().expect("Prefix must exist");
            result.push(code);
            // Add new entry: prefix + last byte.
            dict.insert(w.clone(), next_code);
            next_code = next_code.wrapping_add(1);
            // Reset w to start with the current byte.
            w.clear();
            w.push(last);
        }
        i += 1;
    }
    // Output remaining code.
    if !w.is_empty() {
        let code = dict.get(&w).copied().expect("w must be in dictionary");
        result.push(code);
    }
    result
}

/// Decompresses a sequence of LZW codes back into the original byte data.
///
/// # Parameters
///
/// - `codes`: a slice of u16 codes produced by the `compress` function.
///
/// # Returns
///
/// A vector of bytes representing the decompressed data.
///
/// # Details
///
/// The dictionary is initialized with all 256 single-byte sequences.
/// Then, for each code read, the corresponding dictionary entry is output.
/// The dictionary is updated by appending the first byte of the current entry
/// to the previous entry.
///
/// # Example
///
/// ```
/// use algos::cs::compression::lzw::{compress, decompress};
///
/// let input = b"TOBEORNOTTOBE";
/// let compressed = compress(input);
/// let decompressed = decompress(&compressed);
/// assert_eq!(decompressed, input);
/// ```
pub fn decompress(codes: &[u16]) -> Vec<u8> {
    // Initialize the dictionary with all 256 single-byte sequences.
    let mut dict: Vec<Vec<u8>> = (0..256).map(|i| vec![i as u8]).collect();
    let mut result = Vec::new();

    // Handle first code.
    let mut w = if let Some(&first_code) = codes.first() {
        dict[first_code as usize].clone()
    } else {
        return result;
    };
    result.extend_from_slice(&w);

    for &code in &codes[1..] {
        let entry = match (code as usize).cmp(&dict.len()) {
            std::cmp::Ordering::Less => dict[code as usize].clone(),
            std::cmp::Ordering::Equal => {
                // Special case: code equals dictionary size.
                let mut temp = w.clone();
                temp.push(w[0]);
                temp
            }
            std::cmp::Ordering::Greater => panic!("Bad compressed code: {}", code),
        };
        result.extend_from_slice(&entry);
        // Add new dictionary entry: w + first byte of entry.
        let mut new_entry = w.clone();
        new_entry.push(entry[0]);
        dict.push(new_entry);
        w = entry;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input() {
        let input = b"";
        let compressed = compress(input);
        let decompressed = decompress(&compressed);
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_single_character() {
        let input = b"AAAAAA";
        let compressed = compress(input);
        let decompressed = decompress(&compressed);
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_simple_string() {
        let input = b"TOBEORNOTTOBE";
        let compressed = compress(input);
        let decompressed = decompress(&compressed);
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_longer_input() {
        let input = b"abracadabra abracadabra abracadabra";
        let compressed = compress(input);
        let decompressed = decompress(&compressed);
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_non_ascii() {
        let input = "这是一段测试".as_bytes();
        let compressed = compress(input);
        let decompressed = decompress(&compressed);
        assert_eq!(decompressed, input);
    }
}
