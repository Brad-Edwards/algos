use std::f64;

/// Represents a 2D convolution layer
#[derive(Debug, Clone)]
pub struct Conv2D {
    /// Number of input channels
    pub in_channels: usize,
    /// Number of output channels (number of filters)
    pub out_channels: usize,
    /// Kernel size (height, width)
    pub kernel_size: (usize, usize),
    /// Stride (height, width)
    pub stride: (usize, usize),
    /// Padding (height, width)
    pub padding: (usize, usize),
    /// Weights (out_channels, in_channels, kernel_height, kernel_width)
    pub weights: Vec<Vec<Vec<Vec<f64>>>>,
    /// Biases (out_channels)
    pub bias: Vec<f64>,
}

impl Conv2D {
    /// Creates a new Conv2D layer
    ///
    /// # Arguments
    ///
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Size of the convolving kernel (height, width)
    /// * `stride` - Stride of the convolution (height, width)
    /// * `padding` - Zero-padding added to both sides of the input (height, width)
    ///
    /// # Example
    ///
    /// ```
    /// use algos::ml::deep::conv::Conv2D;
    /// let conv = Conv2D::new(3, 64, (3, 3), (1, 1), (1, 1));
    /// ```
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        // Initialize weights with Kaiming initialization
        let scale = (2.0 / (in_channels * kernel_size.0 * kernel_size.1) as f64).sqrt();
        let weights = (0..out_channels)
            .map(|_| {
                (0..in_channels)
                    .map(|_| {
                        (0..kernel_size.0)
                            .map(|_| {
                                (0..kernel_size.1)
                                    .map(|_| (rand::random::<f64>() * 2.0 - 1.0) * scale)
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let bias = vec![0.0; out_channels];

        Conv2D {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            weights,
            bias,
        }
    }

    /// Computes the output shape for given input dimensions
    ///
    /// # Arguments
    ///
    /// * `input_height` - Height of input
    /// * `input_width` - Width of input
    ///
    /// # Returns
    ///
    /// * Tuple of (output_height, output_width)
    pub fn output_shape(&self, input_height: usize, input_width: usize) -> (usize, usize) {
        let output_height =
            (input_height + 2 * self.padding.0 - self.kernel_size.0) / self.stride.0 + 1;
        let output_width =
            (input_width + 2 * self.padding.1 - self.kernel_size.1) / self.stride.1 + 1;
        (output_height, output_width)
    }

    /// Forward pass of the convolution layer
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape (batch_size, in_channels, height, width)
    ///
    /// # Returns
    ///
    /// * Output tensor of shape (batch_size, out_channels, output_height, output_width)
    /// * Cache for backward pass
    pub fn forward(&self, input: &[Vec<Vec<Vec<f64>>>]) -> (Vec<Vec<Vec<Vec<f64>>>>, Conv2DCache) {
        let batch_size = input.len();
        let input_height = input[0][0].len();
        let input_width = input[0][0][0].len();
        let (output_height, output_width) = self.output_shape(input_height, input_width);

        // Initialize output tensor
        let mut output =
            vec![vec![vec![vec![0.0; output_width]; output_height]; self.out_channels]; batch_size];

        // Pad input if necessary
        let padded_input = if self.padding != (0, 0) {
            self.pad_input(input)
        } else {
            input.to_owned()
        };

        // Perform convolution
        for b in 0..batch_size {
            for out_c in 0..self.out_channels {
                for h in 0..output_height {
                    for w in 0..output_width {
                        let h_start = h * self.stride.0;
                        let w_start = w * self.stride.1;

                        let mut sum = self.bias[out_c];

                        for in_c in 0..self.in_channels {
                            for kh in 0..self.kernel_size.0 {
                                for kw in 0..self.kernel_size.1 {
                                    sum += padded_input[b][in_c][h_start + kh][w_start + kw]
                                        * self.weights[out_c][in_c][kh][kw];
                                }
                            }
                        }

                        output[b][out_c][h][w] = sum;
                    }
                }
            }
        }

        let cache = Conv2DCache {
            input: input.to_owned(),
            padded_input,
            output: output.clone(),
        };

        (output, cache)
    }

    /// Backward pass of the convolution layer
    ///
    /// # Arguments
    ///
    /// * `grad_output` - Gradient of the loss with respect to the output
    /// * `cache` - Cache from forward pass
    ///
    /// # Returns
    ///
    /// * Gradient with respect to input
    /// * Gradient with respect to weights
    /// * Gradient with respect to bias
    pub fn backward(
        &self,
        grad_output: &[Vec<Vec<Vec<f64>>>],
        cache: &Conv2DCache,
    ) -> (Vec<Vec<Vec<Vec<f64>>>>, Conv2DGradients) {
        let batch_size = grad_output.len();
        let (output_height, output_width) = (grad_output[0][0].len(), grad_output[0][0][0].len());

        // Initialize gradients
        let mut grad_input =
            vec![
                vec![
                    vec![vec![0.0; cache.input[0][0][0].len()]; cache.input[0][0].len()];
                    self.in_channels
                ];
                batch_size
            ];
        let mut grad_weights =
            vec![
                vec![vec![vec![0.0; self.kernel_size.1]; self.kernel_size.0]; self.in_channels];
                self.out_channels
            ];
        let mut grad_bias = vec![0.0; self.out_channels];

        // Compute gradients
        for b in 0..batch_size {
            for out_c in 0..self.out_channels {
                for h in 0..output_height {
                    for w in 0..output_width {
                        let h_start = h * self.stride.0;
                        let w_start = w * self.stride.1;

                        // Gradient with respect to bias
                        grad_bias[out_c] += grad_output[b][out_c][h][w];

                        // Gradient with respect to weights
                        for in_c in 0..self.in_channels {
                            for kh in 0..self.kernel_size.0 {
                                for kw in 0..self.kernel_size.1 {
                                    grad_weights[out_c][in_c][kh][kw] += cache.padded_input[b]
                                        [in_c][h_start + kh][w_start + kw]
                                        * grad_output[b][out_c][h][w];

                                    // Gradient with respect to input
                                    if h_start + kh < grad_input[0][0].len()
                                        && w_start + kw < grad_input[0][0][0].len()
                                    {
                                        grad_input[b][in_c][h_start + kh][w_start + kw] += self
                                            .weights[out_c][in_c][kh][kw]
                                            * grad_output[b][out_c][h][w];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let gradients = Conv2DGradients {
            weights: grad_weights,
            bias: grad_bias,
        };

        (grad_input, gradients)
    }

    /// Helper function to pad input tensor
    fn pad_input(&self, input: &[Vec<Vec<Vec<f64>>>]) -> Vec<Vec<Vec<Vec<f64>>>> {
        let batch_size = input.len();
        let input_height = input[0][0].len();
        let input_width = input[0][0][0].len();
        let padded_height = input_height + 2 * self.padding.0;
        let padded_width = input_width + 2 * self.padding.1;

        let mut padded =
            vec![vec![vec![vec![0.0; padded_width]; padded_height]; self.in_channels]; batch_size];

        for b in 0..batch_size {
            for c in 0..self.in_channels {
                for h in 0..input_height {
                    for w in 0..input_width {
                        padded[b][c][h + self.padding.0][w + self.padding.1] = input[b][c][h][w];
                    }
                }
            }
        }

        padded
    }
}

/// Cache for Conv2D forward pass
#[derive(Debug, Clone)]
pub struct Conv2DCache {
    /// Original input
    pub input: Vec<Vec<Vec<Vec<f64>>>>,
    /// Padded input
    pub padded_input: Vec<Vec<Vec<Vec<f64>>>>,
    /// Output
    pub output: Vec<Vec<Vec<Vec<f64>>>>,
}

/// Gradients for Conv2D parameters
#[derive(Debug, Clone)]
pub struct Conv2DGradients {
    /// Gradients for weights
    pub weights: Vec<Vec<Vec<Vec<f64>>>>,
    /// Gradients for bias
    pub bias: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests Conv2D initialization
    #[test]
    fn test_conv2d_initialization() {
        let conv = Conv2D::new(3, 64, (3, 3), (1, 1), (1, 1));
        assert_eq!(conv.in_channels, 3);
        assert_eq!(conv.out_channels, 64);
        assert_eq!(conv.kernel_size, (3, 3));
        assert_eq!(conv.stride, (1, 1));
        assert_eq!(conv.padding, (1, 1));
        assert_eq!(conv.weights.len(), 64);
        assert_eq!(conv.weights[0].len(), 3);
        assert_eq!(conv.weights[0][0].len(), 3);
        assert_eq!(conv.weights[0][0][0].len(), 3);
        assert_eq!(conv.bias.len(), 64);
    }

    /// Tests output shape calculation
    #[test]
    fn test_output_shape() {
        let conv = Conv2D::new(3, 64, (3, 3), (1, 1), (1, 1));
        let (h, w) = conv.output_shape(32, 32);
        assert_eq!(h, 32);
        assert_eq!(w, 32);

        let conv = Conv2D::new(3, 64, (3, 3), (2, 2), (1, 1));
        let (h, w) = conv.output_shape(32, 32);
        assert_eq!(h, 16);
        assert_eq!(w, 16);
    }

    /// Tests forward pass
    #[test]
    fn test_forward() {
        let conv = Conv2D::new(3, 64, (3, 3), (1, 1), (1, 1));
        let input = vec![vec![vec![vec![1.0; 32]; 32]; 3]; 1];
        let (_output, cache) = conv.forward(&input);

        assert_eq!(cache.input, input);
    }

    /// Tests backward pass
    #[test]
    fn test_backward() {
        let conv = Conv2D::new(3, 64, (3, 3), (1, 1), (1, 1));
        let input = vec![vec![vec![vec![1.0; 32]; 32]; 3]; 1];
        let (_output, cache) = conv.forward(&input);

        let grad_output = vec![vec![vec![vec![1.0; 32]; 32]; 64]; 1];
        let (grad_input, gradients) = conv.backward(&grad_output, &cache);

        assert_eq!(grad_input.len(), 1);
        assert_eq!(grad_input[0].len(), 3);
        assert_eq!(grad_input[0][0].len(), 32);
        assert_eq!(grad_input[0][0][0].len(), 32);

        assert_eq!(gradients.weights.len(), 64);
        assert_eq!(gradients.weights[0].len(), 3);
        assert_eq!(gradients.weights[0][0].len(), 3);
        assert_eq!(gradients.weights[0][0][0].len(), 3);
        assert_eq!(gradients.bias.len(), 64);
    }
}
