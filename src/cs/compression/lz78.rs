use std::collections::HashMap;

/// An LZ78 token.
///
/// Each token is a pair (index, next), where:
/// - `index` is the dictionary index of the longest previously seen phrase that is a prefix of the current input.
/// - `next` is the next byte that did not match (or `None` if the input ended exactly).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub index: usize,
    pub next: Option<u8>,
}

/// Compress input data using the LZ78 algorithm.
///
/// # Algorithm
///
/// 1. Start with a dictionary containing only the empty string at index 0.
/// 2. For each position in the input, find the longest phrase `w` (possibly empty)
///    that is already in the dictionary.
/// 3. Let `c` be the next byte (if any). Output the token `(index(w), c)`.
/// 4. Insert the phrase `w+c` into the dictionary at the next available index.
/// 5. Continue until the entire input is processed. If the input ends exactly on a phrase,
///    output a token with `next = None`.
///
/// # Parameters
///
/// - `input`: A slice of bytes to compress.
///
/// # Returns
///
/// A vector of `Token` representing the compressed data.
///
/// # Example
///
/// ```
/// use algos::cs::compression::lz78::compress;
///
/// let data = b"TOBEORNOTTOBE";
/// let tokens = compress(data);
/// assert!(!tokens.is_empty());
/// ```
pub fn compress(input: &[u8]) -> Vec<Token> {
    // Dictionary mapping phrases (as Vec<u8>) to their dictionary index.
    // Start with the empty phrase at index 0.
    let mut dict: HashMap<Vec<u8>, usize> = HashMap::new();
    dict.insert(Vec::new(), 0);
    let mut next_index = 1;
    let mut tokens = Vec::new();

    let mut i = 0;
    while i < input.len() {
        let mut current = Vec::new();
        let mut index = 0; // dictionary index for current phrase (initially empty).
                           // Extend current phrase as long as (current + next byte) exists in dictionary.
        while i < input.len() {
            current.push(input[i]);
            if let Some(&idx) = dict.get(&current) {
                index = idx;
                i += 1;
            } else {
                // The new phrase (current) is not in dictionary.
                break;
            }
        }
        // If we have not reached end-of-input, then the last byte in current is the new byte.
        if i < input.len() {
            // Remove the last byte from current to get the longest phrase that existed.
            let mut phrase = current.clone();
            phrase.pop();
            // index already holds dictionary index for phrase.
            let next_byte = input[i];
            tokens.push(Token {
                index,
                next: Some(next_byte),
            });
            // Insert new phrase = phrase + next_byte into dictionary.
            let mut new_phrase = phrase;
            new_phrase.push(next_byte);
            dict.insert(new_phrase, next_index);
            next_index += 1;
            i += 1;
        } else {
            // End of input reached exactly.
            tokens.push(Token { index, next: None });
        }
    }
    tokens
}

/// Decompress a sequence of LZ78 tokens back into the original data.
///
/// # Algorithm
///
/// 1. Initialize the dictionary with the empty phrase at index 0.
/// 2. For each token (index, next), let phrase = dictionary[index] concatenated with next (if any).
/// 3. Append phrase to the output and add it to the dictionary.
///
/// # Parameters
///
/// - `tokens`: A slice of tokens produced by `compress`.
///
/// # Returns
///
/// A vector of bytes representing the decompressed data.
///
/// # Example
///
/// ```
/// use algos::cs::compression::lz78::{compress, decompress};
///
/// let data = b"TOBEORNOTTOBE";
/// let tokens = compress(data);
/// let decompressed = decompress(&tokens);
/// assert_eq!(decompressed, data);
/// ```
pub fn decompress(tokens: &[Token]) -> Vec<u8> {
    let mut dict: Vec<Vec<u8>> = Vec::new();
    dict.push(Vec::new()); // index 0: empty phrase
    let mut output = Vec::new();

    for token in tokens {
        let mut phrase = dict[token.index].clone();
        if let Some(b) = token.next {
            phrase.push(b);
        }
        output.extend_from_slice(&phrase);
        dict.push(phrase);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input() {
        let input = b"";
        let tokens = compress(input);
        // For empty input, expect no tokens.
        assert!(tokens.is_empty());
        let decompressed = decompress(&tokens);
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_single_character() {
        let input = b"AAAAAA";
        let tokens = compress(input);
        // For a repeated character, the tokens should compress well.
        // Example: [ (0, 'A'), (1, None) ] or similar.
        let decompressed = decompress(&tokens);
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_simple_string() {
        let input = b"TOBEORNOTTOBE";
        let tokens = compress(input);
        let decompressed = decompress(&tokens);
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_longer_input() {
        let input = b"abracadabra abracadabra abracadabra";
        let tokens = compress(input);
        let decompressed = decompress(&tokens);
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_non_ascii() {
        let input = "这是一段测试".as_bytes();
        let tokens = compress(input);
        let decompressed = decompress(&tokens);
        assert_eq!(decompressed, input);
    }
}
