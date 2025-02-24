use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// Represents a node in the Huffman tree.
#[derive(Debug, Clone)]
pub enum HuffmanNode {
    /// A leaf node contains a character and its frequency.
    Leaf { ch: char, freq: usize },
    /// An internal node with left and right children and combined frequency.
    Internal {
        freq: usize,
        left: Box<HuffmanNode>,
        right: Box<HuffmanNode>,
    },
}

impl PartialEq for HuffmanNode {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Leaf {
                    ch: ch1,
                    freq: freq1,
                },
                Self::Leaf {
                    ch: ch2,
                    freq: freq2,
                },
            ) => ch1 == ch2 && freq1 == freq2,
            (
                Self::Internal {
                    freq: freq1,
                    left: left1,
                    right: right1,
                },
                Self::Internal {
                    freq: freq2,
                    left: left2,
                    right: right2,
                },
            ) => freq1 == freq2 && left1 == left2 && right1 == right2,
            _ => false,
        }
    }
}

impl Eq for HuffmanNode {}

impl HuffmanNode {
    /// Returns the frequency of the node.
    pub fn freq(&self) -> usize {
        match self {
            HuffmanNode::Leaf { freq, .. } => *freq,
            HuffmanNode::Internal { freq, .. } => *freq,
        }
    }
}

/// A helper wrapper for HuffmanNode for use in a BinaryHeap.
/// We want the node with the smallest frequency to have highest priority.
#[derive(Debug, Clone, Eq, PartialEq)]
struct NodeWrapper(Box<HuffmanNode>);

impl Ord for NodeWrapper {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse: lower frequency should come first.
        other.0.freq().cmp(&self.0.freq())
    }
}

impl PartialOrd for NodeWrapper {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Build a frequency table mapping each character in `input` to its frequency.
pub fn build_frequency_table(input: &str) -> HashMap<char, usize> {
    let mut freq = HashMap::new();
    for ch in input.chars() {
        *freq.entry(ch).or_insert(0) += 1;
    }
    freq
}

/// Build the Huffman tree given a frequency table.
/// Returns `None` if the frequency table is empty.
pub fn build_huffman_tree(freq_table: &HashMap<char, usize>) -> Option<Box<HuffmanNode>> {
    let mut heap = BinaryHeap::new();
    // Create a leaf node for each character and push it into the heap.
    for (&ch, &freq) in freq_table.iter() {
        heap.push(NodeWrapper(Box::new(HuffmanNode::Leaf { ch, freq })));
    }
    if heap.is_empty() {
        return None;
    }
    // Combine nodes until only one tree remains.
    while heap.len() > 1 {
        let NodeWrapper(left) = heap.pop().unwrap();
        let NodeWrapper(right) = heap.pop().unwrap();
        let combined_freq = left.freq() + right.freq();
        let internal = Box::new(HuffmanNode::Internal {
            freq: combined_freq,
            left,
            right,
        });
        heap.push(NodeWrapper(internal));
    }
    Some(heap.pop().unwrap().0)
}

/// Recursively build the code table mapping characters to their Huffman codes.
///
/// If the tree consists of a single leaf (i.e. one unique symbol), the code "0" is assigned.
pub fn build_code_table(node: &HuffmanNode) -> HashMap<char, String> {
    let mut table = HashMap::new();
    build_code_table_helper(node, String::new(), &mut table);
    table
}

fn build_code_table_helper(node: &HuffmanNode, prefix: String, table: &mut HashMap<char, String>) {
    match node {
        HuffmanNode::Leaf { ch, .. } => {
            let code = if prefix.is_empty() {
                "0".to_string()
            } else {
                prefix
            };
            table.insert(*ch, code);
        }
        HuffmanNode::Internal { left, right, .. } => {
            let mut left_prefix = prefix.clone();
            left_prefix.push('0');
            build_code_table_helper(left, left_prefix, table);
            let mut right_prefix = prefix;
            right_prefix.push('1');
            build_code_table_helper(right, right_prefix, table);
        }
    }
}

/// Encode the input string using the provided code table.
/// Each character is replaced with its Huffman code.
pub fn encode(input: &str, code_table: &HashMap<char, String>) -> String {
    input
        .chars()
        .map(|ch| code_table.get(&ch).unwrap().clone())
        .collect()
}

/// Decode an encoded bit string using the Huffman tree.
/// Traverses the tree according to each bit until a leaf is reached.
pub fn decode(encoded: &str, tree: &HuffmanNode) -> String {
    let mut result = String::new();
    let mut current = tree;

    // Special case: if tree is a leaf, each '0' represents one occurrence
    if let HuffmanNode::Leaf { ch, .. } = tree {
        return encoded.chars().map(|_| *ch).collect();
    }

    for bit in encoded.chars() {
        if let HuffmanNode::Internal { left, right, .. } = current {
            current = if bit == '0' { left } else { right };
            if let HuffmanNode::Leaf { ch, .. } = current {
                result.push(*ch);
                current = tree;
            }
        }
    }

    result
}

/// Convenience function: builds the Huffman tree from input, encodes the input,
/// and returns (encoded bit string, Huffman tree).
pub fn huffman_encode(input: &str) -> (String, Box<HuffmanNode>) {
    let freq_table = build_frequency_table(input);
    let tree = build_huffman_tree(&freq_table).expect("Input must be non-empty");
    let code_table = build_code_table(&tree);
    let encoded = encode(input, &code_table);
    (encoded, tree)
}

/// Convenience function: decodes an encoded bit string using the provided Huffman tree.
pub fn huffman_decode(encoded: &str, tree: &HuffmanNode) -> String {
    decode(encoded, tree)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frequency_table() {
        let input = "aabccc";
        let freq = build_frequency_table(input);
        assert_eq!(freq.get(&'a'), Some(&2));
        assert_eq!(freq.get(&'b'), Some(&1));
        assert_eq!(freq.get(&'c'), Some(&3));
    }

    #[test]
    fn test_huffman_tree_and_code_table() {
        let input = "this is an example for huffman encoding";
        let freq = build_frequency_table(input);
        let tree = build_huffman_tree(&freq).expect("Tree should be built");
        let code_table = build_code_table(&tree);
        // Each character in input must have a code.
        for ch in input.chars() {
            assert!(code_table.contains_key(&ch), "Missing code for '{}'", ch);
        }
    }

    #[test]
    fn test_encode_decode() {
        let input = "huffman coding in rust is fun!";
        let (encoded, tree) = huffman_encode(input);
        let decoded = huffman_decode(&encoded, &tree);
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_single_character() {
        let input = "aaaaaaa";
        let (encoded, tree) = huffman_encode(input);
        // With a single symbol, the assigned code is "0" for each occurrence.
        assert_eq!(encoded, "0".repeat(input.len()));
        let decoded = huffman_decode(&encoded, &tree);
        assert_eq!(decoded, input);
    }
}
