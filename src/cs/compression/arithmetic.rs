use std::vec::Vec;

/// The total number of symbols: 256 bytes plus one EOF symbol.
pub const ALPHABET_SIZE: usize = 257;

/// A frequency model built from an input message.
/// Frequencies are stored for symbols 0..255 and for EOF at index 256.
/// cum_freq holds cumulative frequencies with cum_freq[0] = 0 and
/// cum_freq[ALPHABET_SIZE] = total frequency.
#[derive(Debug, Clone)]
pub struct FrequencyModel {
    pub freq: Vec<u32>,     // length = ALPHABET_SIZE
    pub cum_freq: Vec<u32>, // length = ALPHABET_SIZE + 1
    pub total: u32,
}

impl FrequencyModel {
    /// Build a frequency model from the input data.
    /// Each byte's frequency is counted and an EOF symbol (index 256) is added with frequency 1.
    pub fn new(input: &[u8]) -> Self {
        // Start with frequency 1 for each symbol to avoid zero probabilities
        let mut freq = vec![1u32; ALPHABET_SIZE];

        // Count frequencies for input bytes
        for &b in input {
            freq[b as usize] = freq[b as usize].saturating_add(1);
        }

        // Calculate cumulative frequencies
        let mut cum_freq = vec![0u32; ALPHABET_SIZE + 1];
        let mut total = 0u32;
        for i in 0..ALPHABET_SIZE {
            total = total.saturating_add(freq[i]);
            cum_freq[i + 1] = total;
        }

        // Scale frequencies more aggressively
        while total >= FIRST_QUARTER {
            total = 0;
            for f in freq.iter_mut() {
                *f = (*f + 1) >> 1; // Divide by 2 rounding up
                total = total.saturating_add(*f);
            }
            cum_freq[0] = 0;
            for i in 0..ALPHABET_SIZE {
                cum_freq[i + 1] = cum_freq[i].saturating_add(freq[i]);
            }
        }

        FrequencyModel {
            freq,
            cum_freq,
            total,
        }
    }
}

// Fixed-point parameters for arithmetic coding
const CODE_VALUE_BITS: u32 = 14;
const TOP_VALUE: u32 = (1 << CODE_VALUE_BITS) - 1; // 16383
const FIRST_QUARTER: u32 = (TOP_VALUE + 1) / 4; // 4096
const HALF: u32 = 2 * FIRST_QUARTER; // 8192
const THIRD_QUARTER: u32 = 3 * FIRST_QUARTER; // 12288

/// A simple bit writer that buffers bits into bytes.
struct BitWriter {
    buffer: Vec<u8>,
    current_byte: u8,
    bits_filled: u8,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter {
            buffer: Vec::new(),
            current_byte: 0,
            bits_filled: 0,
        }
    }

    /// Write a single bit (0 or 1).
    fn write_bit(&mut self, bit: u8) {
        // Shift left by 1 and add new bit
        self.current_byte = ((self.current_byte as u16) << 1 | (bit & 1) as u16) as u8;
        self.bits_filled += 1;
        if self.bits_filled == 8 {
            self.buffer.push(self.current_byte);
            self.current_byte = 0;
            self.bits_filled = 0;
        }
    }

    /// Flush remaining bits by padding with zeros.
    fn finish(mut self) -> Vec<u8> {
        if self.bits_filled > 0 {
            // Left shift remaining bits to align with MSB
            self.current_byte = ((self.current_byte as u16) << (8 - self.bits_filled)) as u8;
            self.buffer.push(self.current_byte);
        }
        self.buffer
    }
}

/// A simple bit reader that reads bits from a byte slice.
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    current_byte: u8,
    bits_left: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        let mut reader = BitReader {
            data,
            pos: 0,
            current_byte: 0,
            bits_left: 0,
        };
        reader.read_next_byte();
        reader
    }

    fn read_next_byte(&mut self) {
        if self.pos < self.data.len() {
            self.current_byte = self.data[self.pos];
            self.pos += 1;
            self.bits_left = 8;
        } else {
            self.current_byte = 0;
            self.bits_left = 8;
        }
    }

    /// Read a single bit (MSB first).
    fn read_bit(&mut self) -> u8 {
        if self.bits_left == 0 {
            self.read_next_byte();
        }
        self.bits_left -= 1;
        (self.current_byte >> self.bits_left) & 1
    }
}

/// Encode the input data using arithmetic coding.
///
/// This function uses the provided frequency model to encode the input data.
/// An EOF symbol is automatically appended to mark the end of the message.
///
/// # Example
///
/// ```
/// use algos::cs::compression::arithmetic::{FrequencyModel, arithmetic_encode, arithmetic_decode};
///
/// let input = b"hello arithmetic coding";
/// let model = FrequencyModel::new(input);
/// let encoded = arithmetic_encode(input, &model);
/// let decoded = arithmetic_decode(&encoded, &model);
/// assert_eq!(decoded, input);
/// ```
pub fn arithmetic_encode(input: &[u8], model: &FrequencyModel) -> Vec<u8> {
    let mut bw = BitWriter::new();
    let mut low: u32 = 0;
    let mut high: u32 = TOP_VALUE;
    let mut pending_bits: u32 = 0;

    let mut symbols = input.iter().map(|&b| b as usize).collect::<Vec<usize>>();
    symbols.push(256); // EOF symbol

    for &sym in &symbols {
        let range = high - low + 1;
        let total = model.total;
        let cum_low = model.cum_freq[sym];
        let cum_high = model.cum_freq[sym + 1];

        high = low + (range * cum_high) / total - 1;
        low = low + (range * cum_low) / total;

        loop {
            if high < HALF {
                bw.write_bit(0);
                while pending_bits > 0 {
                    bw.write_bit(1);
                    pending_bits -= 1;
                }
                low <<= 1;
                high = (high << 1) | 1;
            } else if low >= HALF {
                bw.write_bit(1);
                while pending_bits > 0 {
                    bw.write_bit(0);
                    pending_bits -= 1;
                }
                low = (low - HALF) << 1;
                high = ((high - HALF) << 1) | 1;
            } else if low >= FIRST_QUARTER && high < THIRD_QUARTER {
                pending_bits += 1;
                low = (low - FIRST_QUARTER) << 1;
                high = ((high - FIRST_QUARTER) << 1) | 1;
            } else {
                break;
            }
        }
    }

    // Final bit flush so decoder sees where the last symbol ended
    pending_bits += 1;
    if low < FIRST_QUARTER {
        bw.write_bit(0);
        while pending_bits > 0 {
            bw.write_bit(1);
            pending_bits -= 1;
        }
    } else {
        bw.write_bit(1);
        while pending_bits > 0 {
            bw.write_bit(0);
            pending_bits -= 1;
        }
    }

    bw.finish()
}

/// Decode an arithmetic-coded bitstream into the original data.
///
/// The model must be the same as that used for encoding.
/// Decoding continues until the EOF symbol (256) is encountered.
///
/// # Example
///
/// ```
/// use algos::cs::compression::arithmetic::{FrequencyModel, arithmetic_encode, arithmetic_decode};
///
/// let input = b"hello arithmetic coding";
/// let model = FrequencyModel::new(input);
/// let encoded = arithmetic_encode(input, &model);
/// let decoded = arithmetic_decode(&encoded, &model);
/// assert_eq!(decoded, input);
/// ```
pub fn arithmetic_decode(encoded: &[u8], model: &FrequencyModel) -> Vec<u8> {
    let mut br = BitReader::new(encoded);
    let mut low: u32 = 0;
    let mut high: u32 = TOP_VALUE;

    let mut value: u32 = 0;
    for _ in 0..CODE_VALUE_BITS {
        value = (value << 1) | (br.read_bit() as u32);
    }

    let mut output = Vec::new();
    loop {
        let range = high - low + 1;
        let total = model.total;
        let scaled = ((value - low) as u64 * total as u64) / range as u64;

        // Find symbol using binary search
        let mut left = 0;
        let mut right = ALPHABET_SIZE - 1;
        let mut sym = right;

        while left <= right {
            let mid = (left + right) / 2;
            if (model.cum_freq[mid] as u64) <= scaled && scaled < (model.cum_freq[mid + 1] as u64) {
                sym = mid;
                break;
            } else if scaled < model.cum_freq[mid] as u64 {
                if mid == 0 {
                    sym = 0;
                    break;
                }
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        if sym == 256 {
            break;
        }

        output.push(sym as u8);

        let cum_low = model.cum_freq[sym] as u64;
        let cum_high = model.cum_freq[sym + 1] as u64;

        high = (low as u64 + (range as u64 * cum_high) / total as u64 - 1) as u32;
        low = (low as u64 + (range as u64 * cum_low) / total as u64) as u32;

        loop {
            if high < HALF {
                low <<= 1;
                high = (high << 1) | 1;
                value = (value << 1) | (br.read_bit() as u32);
            } else if low >= HALF {
                low = (low - HALF) << 1;
                high = ((high - HALF) << 1) | 1;
                value = ((value - HALF) << 1) | (br.read_bit() as u32);
            } else if low >= FIRST_QUARTER && high < THIRD_QUARTER {
                low = (low - FIRST_QUARTER) << 1;
                high = ((high - FIRST_QUARTER) << 1) | 1;
                value = ((value - FIRST_QUARTER) << 1) | (br.read_bit() as u32);
            } else {
                break;
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frequency_model() {
        let input = b"abracadabra";
        let model = FrequencyModel::new(input);
        // Check that all frequencies are at least 1
        assert!(model.freq.iter().all(|&f| f >= 1));
        // Total should be sum of all frequencies
        let sum: u32 = model.freq.iter().sum();
        assert_eq!(model.total, sum);
        // Check cumulative frequencies are monotonically increasing
        for i in 1..=ALPHABET_SIZE {
            assert!(model.cum_freq[i] > model.cum_freq[i - 1]);
        }
    }

    #[test]
    fn test_encode_decode_empty() {
        let input = b"";
        let model = FrequencyModel::new(input);
        let encoded = arithmetic_encode(input, &model);
        let decoded = arithmetic_decode(&encoded, &model);
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_encode_decode_simple() {
        let input = b"hello arithmetic coding";
        let model = FrequencyModel::new(input);
        let encoded = arithmetic_encode(input, &model);
        let decoded = arithmetic_decode(&encoded, &model);
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_encode_decode_longer() {
        let input = b"The quick brown fox jumps over the lazy dog. Arithmetic coding is cool!";
        let model = FrequencyModel::new(input);
        let encoded = arithmetic_encode(input, &model);
        let decoded = arithmetic_decode(&encoded, &model);
        assert_eq!(decoded, input);
    }
}
