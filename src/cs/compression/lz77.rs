/// An LZ77 token.
///
/// When a match is found, the token is:
///   (offset, length, next)
///
/// If no match is found, then offset and length are zero, and `next` is the literal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub offset: usize,
    pub length: usize,
    pub next: Option<u8>,
}

/// Compress the input data using the LZ77 algorithm.
///
/// # Parameters
///
/// - `input`: the data to compress as a byte slice.
/// - `window_size`: the maximum number of previous bytes to search for a match.
/// - `lookahead_buffer_size`: the maximum match length to consider.
///
/// # Returns
///
/// A vector of `Token` representing the compressed data.
///
/// # Example
///
/// ```
/// use algos::cs::compression::lz77::compress;
///
/// let data = b"abracadabra abracadabra";
/// let tokens = compress(data, 16, 8);
/// assert!(!tokens.is_empty());
/// ```
pub fn compress(input: &[u8], window_size: usize, lookahead_buffer_size: usize) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < input.len() {
        let end = input.len();
        let search_start = i.saturating_sub(window_size);
        let mut best_length = 0;
        let mut best_offset = 0;
        // Look for the longest match in the sliding window.
        for j in search_start..i {
            let mut length = 0;
            // Compare input[j..] with input[i..] until mismatch, end of input,
            // or reaching the lookahead buffer limit.
            while i + length < end &&
                  j + length < i &&  // ensure we remain in the window
                  length < lookahead_buffer_size &&
                  input[j + length] == input[i + length]
            {
                length += 1;
            }
            if length > best_length {
                best_length = length;
                best_offset = i - j;
            }
        }
        if best_length > 0 {
            // If the match reaches to the end, then next is None.
            let next = if i + best_length < end {
                Some(input[i + best_length])
            } else {
                None
            };
            tokens.push(Token {
                offset: best_offset,
                length: best_length,
                next,
            });
            i += best_length + 1;
        } else {
            // No match found: output literal token.
            tokens.push(Token {
                offset: 0,
                length: 0,
                next: Some(input[i]),
            });
            i += 1;
        }
    }
    tokens
}

/// Decompress a sequence of LZ77 tokens into the original data.
///
/// # Parameters
///
/// - `tokens`: a slice of `Token` produced by the `compress` function.
///
/// # Returns
///
/// A `Vec<u8>` containing the decompressed data.
///
/// # Example
///
/// ```
/// use algos::cs::compression::lz77::{compress, decompress};
///
/// let data = b"abracadabra abracadabra";
/// let tokens = compress(data, 16, 8);
/// let decompressed = decompress(&tokens);
/// assert_eq!(decompressed, data);
/// ```
pub fn decompress(tokens: &[Token]) -> Vec<u8> {
    let mut output = Vec::new();
    for token in tokens {
        // If length > 0, copy the matching substring from output.
        if token.length > 0 {
            let start = output.len().saturating_sub(token.offset);
            for i in 0..token.length {
                // Since the token was produced correctly, start+i is valid.
                output.push(output[start + i]);
            }
        }
        // Append the literal (if present).
        if let Some(byte) = token.next {
            output.push(byte);
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input() {
        let input = b"";
        let tokens = compress(input, 16, 8);
        let decompressed = decompress(&tokens);
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_no_repetition() {
        // When there is no repetition, every token should be a literal.
        let input = b"abcdefg";
        let tokens = compress(input, 16, 8);
        for token in &tokens {
            assert_eq!(token.length, 0);
            assert_eq!(token.offset, 0);
            assert!(token.next.is_some());
        }
        let decompressed = decompress(&tokens);
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_single_character_repetition() {
        let input = b"aaaaaaa";
        let tokens = compress(input, 16, 8);
        let decompressed = decompress(&tokens);
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_compress_decompress() {
        let input = b"abracadabra abracadabra";
        // Use a moderate window and lookahead sizes.
        let tokens = compress(input, 16, 8);
        let decompressed = decompress(&tokens);
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_longer_input() {
        let input = b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. \
                      Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.";
        let tokens = compress(input, 32, 16);
        let decompressed = decompress(&tokens);
        assert_eq!(decompressed, input);
    }
}
