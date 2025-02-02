use std::f64;

/// Activation functions for RNN cells
#[derive(Debug, Clone, Copy)]
pub enum Activation {
    Tanh,
    Sigmoid,
    ReLU,
}

impl Activation {
    /// Apply the activation function
    fn forward(&self, x: f64) -> f64 {
        match self {
            Activation::Tanh => x.tanh(),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::ReLU => x.max(0.0),
        }
    }

    /// Compute the derivative of the activation function
    fn backward(&self, x: f64) -> f64 {
        match self {
            Activation::Tanh => 1.0 - x.powi(2),
            Activation::Sigmoid => x * (1.0 - x),
            Activation::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
        }
    }
}

/// Basic RNN cell implementation
#[derive(Debug, Clone)]
pub struct RNNCell {
    /// Input dimension
    pub input_size: usize,
    /// Hidden state dimension
    pub hidden_size: usize,
    /// Input weights
    pub w_ih: Vec<Vec<f64>>,
    /// Hidden state weights
    pub w_hh: Vec<Vec<f64>>,
    /// Bias
    pub bias: Vec<f64>,
    /// Activation function
    pub activation: Activation,
}

impl RNNCell {
    /// Creates a new RNN cell with the specified dimensions
    ///
    /// # Arguments
    ///
    /// * `input_size` - Size of input features
    /// * `hidden_size` - Size of hidden state
    /// * `activation` - Activation function to use
    ///
    /// # Example
    ///
    /// ```
    /// use algos::ml::deep::bptt::{RNNCell, Activation};
    /// let rnn = RNNCell::new(10, 20, Activation::Tanh);
    /// ```
    pub fn new(input_size: usize, hidden_size: usize, activation: Activation) -> Self {
        // Initialize weights with small random values
        let w_ih = (0..hidden_size)
            .map(|_| (0..input_size).map(|_| rand::random::<f64>() * 0.01).collect())
            .collect();
        let w_hh = (0..hidden_size)
            .map(|_| (0..hidden_size).map(|_| rand::random::<f64>() * 0.01).collect())
            .collect();
        let bias = vec![0.0; hidden_size];

        RNNCell {
            input_size,
            hidden_size,
            w_ih,
            w_hh,
            bias,
            activation,
        }
    }

    /// Forward pass of the RNN cell
    ///
    /// # Arguments
    ///
    /// * `input` - Input vector
    /// * `hidden` - Previous hidden state
    ///
    /// # Returns
    ///
    /// * New hidden state
    /// * Cache for backward pass
    pub fn forward(&self, input: &[f64], hidden: &[f64]) -> (Vec<f64>, RNNCellCache) {
        assert_eq!(input.len(), self.input_size, "Input size mismatch");
        assert_eq!(hidden.len(), self.hidden_size, "Hidden size mismatch");

        let mut new_hidden = vec![0.0; self.hidden_size];
        let mut pre_activation = vec![0.0; self.hidden_size];

        // Compute linear combination
        for i in 0..self.hidden_size {
            let mut sum = self.bias[i];
            for j in 0..self.input_size {
                sum += self.w_ih[i][j] * input[j];
            }
            for j in 0..self.hidden_size {
                sum += self.w_hh[i][j] * hidden[j];
            }
            pre_activation[i] = sum;
            new_hidden[i] = self.activation.forward(sum);
        }

        let cache = RNNCellCache {
            input: input.to_vec(),
            hidden: hidden.to_vec(),
            pre_activation,
            new_hidden: new_hidden.clone(),
        };

        (new_hidden, cache)
    }

    /// Backward pass of the RNN cell
    ///
    /// # Arguments
    ///
    /// * `grad_next` - Gradient from the next timestep
    /// * `grad_output` - Gradient from the output (if any)
    /// * `cache` - Cache from forward pass
    ///
    /// # Returns
    ///
    /// * Gradient with respect to input
    /// * Gradient with respect to hidden state
    /// * Gradient with respect to parameters (w_ih, w_hh, bias)
    pub fn backward(&self, grad_next: &[f64], grad_output: &[f64], cache: &RNNCellCache) 
        -> (Vec<f64>, Vec<f64>, RNNGradients) 
    {
        let mut grad_input = vec![0.0; self.input_size];
        let mut grad_hidden = vec![0.0; self.hidden_size];
        let mut grad_w_ih = vec![vec![0.0; self.input_size]; self.hidden_size];
        let mut grad_w_hh = vec![vec![0.0; self.hidden_size]; self.hidden_size];
        let mut grad_bias = vec![0.0; self.hidden_size];

        // Combine gradients from next timestep and output
        let mut total_grad = vec![0.0; self.hidden_size];
        for i in 0..self.hidden_size {
            total_grad[i] = grad_next[i] + grad_output[i];
        }

        // Backpropagate through activation
        let mut grad_pre = vec![0.0; self.hidden_size];
        for i in 0..self.hidden_size {
            grad_pre[i] = total_grad[i] * 
                         self.activation.backward(cache.new_hidden[i]);
        }

        // Compute gradients
        for i in 0..self.hidden_size {
            grad_bias[i] = grad_pre[i];
            
            for j in 0..self.input_size {
                grad_w_ih[i][j] = grad_pre[i] * cache.input[j];
                grad_input[j] += grad_pre[i] * self.w_ih[i][j];
            }
            
            for j in 0..self.hidden_size {
                grad_w_hh[i][j] = grad_pre[i] * cache.hidden[j];
                grad_hidden[j] += grad_pre[i] * self.w_hh[i][j];
            }
        }

        let gradients = RNNGradients {
            w_ih: grad_w_ih,
            w_hh: grad_w_hh,
            bias: grad_bias,
        };

        (grad_input, grad_hidden, gradients)
    }
}

/// LSTM cell implementation
#[derive(Debug, Clone)]
pub struct LSTMCell {
    /// Input dimension
    pub input_size: usize,
    /// Hidden state dimension
    pub hidden_size: usize,
    /// Input weights for input gate
    pub w_ii: Vec<Vec<f64>>,
    /// Hidden weights for input gate
    pub w_hi: Vec<Vec<f64>>,
    /// Input weights for forget gate
    pub w_if: Vec<Vec<f64>>,
    /// Hidden weights for forget gate
    pub w_hf: Vec<Vec<f64>>,
    /// Input weights for output gate
    pub w_io: Vec<Vec<f64>>,
    /// Hidden weights for output gate
    pub w_ho: Vec<Vec<f64>>,
    /// Input weights for cell gate
    pub w_ig: Vec<Vec<f64>>,
    /// Hidden weights for cell gate
    pub w_hg: Vec<Vec<f64>>,
    /// Biases for all gates
    pub bias: Vec<f64>,
}

impl LSTMCell {
    /// Creates a new LSTM cell with the specified dimensions
    ///
    /// # Arguments
    ///
    /// * `input_size` - Size of input features
    /// * `hidden_size` - Size of hidden state
    ///
    /// # Example
    ///
    /// ```
    /// use algos::ml::deep::bptt::LSTMCell;
    /// let lstm = LSTMCell::new(10, 20);
    /// ```
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        // Helper function to create weight matrix
        let create_weights = |rows: usize, cols: usize| -> Vec<Vec<f64>> {
            (0..rows)
                .map(|_| (0..cols).map(|_| rand::random::<f64>() * 0.01).collect())
                .collect()
        };

        LSTMCell {
            input_size,
            hidden_size,
            w_ii: create_weights(hidden_size, input_size),
            w_hi: create_weights(hidden_size, hidden_size),
            w_if: create_weights(hidden_size, input_size),
            w_hf: create_weights(hidden_size, hidden_size),
            w_io: create_weights(hidden_size, input_size),
            w_ho: create_weights(hidden_size, hidden_size),
            w_ig: create_weights(hidden_size, input_size),
            w_hg: create_weights(hidden_size, hidden_size),
            bias: vec![0.0; hidden_size * 4], // biases for input, forget, output, and cell gates
        }
    }

    /// Forward pass of the LSTM cell
    ///
    /// # Arguments
    ///
    /// * `input` - Input vector
    /// * `hidden` - Previous hidden state
    /// * `cell` - Previous cell state
    ///
    /// # Returns
    ///
    /// * New hidden state
    /// * New cell state
    /// * Cache for backward pass
    pub fn forward(&self, input: &[f64], hidden: &[f64], cell: &[f64]) 
        -> (Vec<f64>, Vec<f64>, LSTMCache) 
    {
        assert_eq!(input.len(), self.input_size, "Input size mismatch");
        assert_eq!(hidden.len(), self.hidden_size, "Hidden size mismatch");
        assert_eq!(cell.len(), self.hidden_size, "Cell size mismatch");

        let mut gates = vec![0.0; self.hidden_size * 4];
        let sigmoid = Activation::Sigmoid;
        let tanh = Activation::Tanh;

        // Compute gates
        for i in 0..self.hidden_size {
            // Input gate
            let mut input_gate = self.bias[i];
            for j in 0..self.input_size {
                input_gate += self.w_ii[i][j] * input[j];
            }
            for j in 0..self.hidden_size {
                input_gate += self.w_hi[i][j] * hidden[j];
            }
            gates[i] = sigmoid.forward(input_gate);

            // Forget gate
            let mut forget_gate = self.bias[i + self.hidden_size];
            for j in 0..self.input_size {
                forget_gate += self.w_if[i][j] * input[j];
            }
            for j in 0..self.hidden_size {
                forget_gate += self.w_hf[i][j] * hidden[j];
            }
            gates[i + self.hidden_size] = sigmoid.forward(forget_gate);

            // Output gate
            let mut output_gate = self.bias[i + 2 * self.hidden_size];
            for j in 0..self.input_size {
                output_gate += self.w_io[i][j] * input[j];
            }
            for j in 0..self.hidden_size {
                output_gate += self.w_ho[i][j] * hidden[j];
            }
            gates[i + 2 * self.hidden_size] = sigmoid.forward(output_gate);

            // Cell gate
            let mut cell_gate = self.bias[i + 3 * self.hidden_size];
            for j in 0..self.input_size {
                cell_gate += self.w_ig[i][j] * input[j];
            }
            for j in 0..self.hidden_size {
                cell_gate += self.w_hg[i][j] * hidden[j];
            }
            gates[i + 3 * self.hidden_size] = tanh.forward(cell_gate);
        }

        let mut new_cell = vec![0.0; self.hidden_size];
        let mut new_hidden = vec![0.0; self.hidden_size];

        // Compute new cell and hidden states
        for i in 0..self.hidden_size {
            new_cell[i] = gates[i + self.hidden_size] * cell[i] + 
                         gates[i] * gates[i + 3 * self.hidden_size];
            new_hidden[i] = gates[i + 2 * self.hidden_size] * tanh.forward(new_cell[i]);
        }

        let cache = LSTMCache {
            input: input.to_vec(),
            hidden: hidden.to_vec(),
            cell: cell.to_vec(),
            gates,
            new_cell: new_cell.clone(),
            new_hidden: new_hidden.clone(),
        };

        (new_hidden, new_cell, cache)
    }

    /// Backward pass of the LSTM cell
    ///
    /// # Arguments
    ///
    /// * `grad_next_h` - Gradient from the next timestep (hidden)
    /// * `grad_next_c` - Gradient from the next timestep (cell)
    /// * `cache` - Cache from forward pass
    ///
    /// # Returns
    ///
    /// * Gradient with respect to input
    /// * Gradient with respect to hidden state
    /// * Gradient with respect to cell state
    /// * Gradient with respect to parameters
    pub fn backward(&self, grad_next_h: &[f64], grad_next_c: &[f64], cache: &LSTMCache) 
        -> (Vec<f64>, Vec<f64>, Vec<f64>, LSTMGradients) 
    {
        let mut grad_input = vec![0.0; self.input_size];
        let mut grad_hidden = vec![0.0; self.hidden_size];
        let mut grad_cell = vec![0.0; self.hidden_size];
        let mut grads = LSTMGradients::new(self.input_size, self.hidden_size);

        let tanh = Activation::Tanh;
        let sigmoid = Activation::Sigmoid;

        // Backpropagate through the timestep
        for i in 0..self.hidden_size {
            let input_gate = cache.gates[i];
            let forget_gate = cache.gates[i + self.hidden_size];
            let output_gate = cache.gates[i + 2 * self.hidden_size];
            let cell_gate = cache.gates[i + 3 * self.hidden_size];

            // Gradient of hidden state
            let tanh_new_cell = tanh.forward(cache.new_cell[i]);
            let grad_h = grad_next_h[i];
            
            // Gradient of cell state
            let grad_c = grad_next_c[i] + 
                        grad_h * output_gate * tanh.backward(tanh_new_cell);

            // Gradient of gates
            let grad_input_gate = grad_c * cell_gate;
            let grad_forget_gate = grad_c * cache.cell[i];
            let grad_output_gate = grad_h * tanh_new_cell;
            let grad_cell_gate = grad_c * input_gate;

            // Accumulate gradients
            self.accumulate_gradients(
                i, &cache.input, &cache.hidden,
                grad_input_gate, grad_forget_gate, grad_output_gate, grad_cell_gate,
                &mut grad_input, &mut grad_hidden, &mut grads
            );

            grad_cell[i] = grad_c * forget_gate;
        }

        (grad_input, grad_hidden, grad_cell, grads)
    }

    /// Helper method to accumulate gradients for all gates
    fn accumulate_gradients(
        &self,
        idx: usize,
        input: &[f64],
        hidden: &[f64],
        grad_input_gate: f64,
        grad_forget_gate: f64,
        grad_output_gate: f64,
        grad_cell_gate: f64,
        grad_input: &mut [f64],
        grad_hidden: &mut [f64],
        grads: &mut LSTMGradients,
    ) {
        // Input gate gradients
        for j in 0..self.input_size {
            grads.w_ii[idx][j] += grad_input_gate * input[j];
            grad_input[j] += grad_input_gate * self.w_ii[idx][j];
        }
        for j in 0..self.hidden_size {
            grads.w_hi[idx][j] += grad_input_gate * hidden[j];
            grad_hidden[j] += grad_input_gate * self.w_hi[idx][j];
        }

        // Forget gate gradients
        for j in 0..self.input_size {
            grads.w_if[idx][j] += grad_forget_gate * input[j];
            grad_input[j] += grad_forget_gate * self.w_if[idx][j];
        }
        for j in 0..self.hidden_size {
            grads.w_hf[idx][j] += grad_forget_gate * hidden[j];
            grad_hidden[j] += grad_forget_gate * self.w_hf[idx][j];
        }

        // Output gate gradients
        for j in 0..self.input_size {
            grads.w_io[idx][j] += grad_output_gate * input[j];
            grad_input[j] += grad_output_gate * self.w_io[idx][j];
        }
        for j in 0..self.hidden_size {
            grads.w_ho[idx][j] += grad_output_gate * hidden[j];
            grad_hidden[j] += grad_output_gate * self.w_ho[idx][j];
        }

        // Cell gate gradients
        for j in 0..self.input_size {
            grads.w_ig[idx][j] += grad_cell_gate * input[j];
            grad_input[j] += grad_cell_gate * self.w_ig[idx][j];
        }
        for j in 0..self.hidden_size {
            grads.w_hg[idx][j] += grad_cell_gate * hidden[j];
            grad_hidden[j] += grad_cell_gate * self.w_hg[idx][j];
        }

        // Bias gradients
        grads.bias[idx] += grad_input_gate;
        grads.bias[idx + self.hidden_size] += grad_forget_gate;
        grads.bias[idx + 2 * self.hidden_size] += grad_output_gate;
        grads.bias[idx + 3 * self.hidden_size] += grad_cell_gate;
    }
}

/// Cache for RNN cell forward pass
#[derive(Debug, Clone)]
pub struct RNNCellCache {
    /// Input vector
    pub input: Vec<f64>,
    /// Previous hidden state
    pub hidden: Vec<f64>,
    /// Pre-activation values
    pub pre_activation: Vec<f64>,
    /// New hidden state
    pub new_hidden: Vec<f64>,
}

/// Cache for LSTM cell forward pass
#[derive(Debug, Clone)]
pub struct LSTMCache {
    /// Input vector
    pub input: Vec<f64>,
    /// Previous hidden state
    pub hidden: Vec<f64>,
    /// Previous cell state
    pub cell: Vec<f64>,
    /// Gate activations
    pub gates: Vec<f64>,
    /// New cell state
    pub new_cell: Vec<f64>,
    /// New hidden state
    pub new_hidden: Vec<f64>,
}

/// Gradients for RNN parameters
#[derive(Debug, Clone)]
pub struct RNNGradients {
    /// Gradients for input weights
    pub w_ih: Vec<Vec<f64>>,
    /// Gradients for hidden weights
    pub w_hh: Vec<Vec<f64>>,
    /// Gradients for bias
    pub bias: Vec<f64>,
}

/// Gradients for LSTM parameters
#[derive(Debug, Clone)]
pub struct LSTMGradients {
    /// Gradients for input gate weights (input)
    pub w_ii: Vec<Vec<f64>>,
    /// Gradients for input gate weights (hidden)
    pub w_hi: Vec<Vec<f64>>,
    /// Gradients for forget gate weights (input)
    pub w_if: Vec<Vec<f64>>,
    /// Gradients for forget gate weights (hidden)
    pub w_hf: Vec<Vec<f64>>,
    /// Gradients for output gate weights (input)
    pub w_io: Vec<Vec<f64>>,
    /// Gradients for output gate weights (hidden)
    pub w_ho: Vec<Vec<f64>>,
    /// Gradients for cell gate weights (input)
    pub w_ig: Vec<Vec<f64>>,
    /// Gradients for cell gate weights (hidden)
    pub w_hg: Vec<Vec<f64>>,
    /// Gradients for all biases
    pub bias: Vec<f64>,
}

impl LSTMGradients {
    /// Creates a new LSTMGradients instance with zeroed gradients
    fn new(input_size: usize, hidden_size: usize) -> Self {
        let create_zeros = |rows: usize, cols: usize| -> Vec<Vec<f64>> {
            vec![vec![0.0; cols]; rows]
        };

        LSTMGradients {
            w_ii: create_zeros(hidden_size, input_size),
            w_hi: create_zeros(hidden_size, hidden_size),
            w_if: create_zeros(hidden_size, input_size),
            w_hf: create_zeros(hidden_size, hidden_size),
            w_io: create_zeros(hidden_size, input_size),
            w_ho: create_zeros(hidden_size, hidden_size),
            w_ig: create_zeros(hidden_size, input_size),
            w_hg: create_zeros(hidden_size, hidden_size),
            bias: vec![0.0; hidden_size * 4],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests RNN cell initialization
    #[test]
    fn test_rnn_cell_initialization() {
        let rnn = RNNCell::new(3, 4, Activation::Tanh);
        assert_eq!(rnn.input_size, 3);
        assert_eq!(rnn.hidden_size, 4);
        assert_eq!(rnn.w_ih.len(), 4);
        assert_eq!(rnn.w_ih[0].len(), 3);
        assert_eq!(rnn.w_hh.len(), 4);
        assert_eq!(rnn.w_hh[0].len(), 4);
        assert_eq!(rnn.bias.len(), 4);
    }

    /// Tests RNN forward pass
    #[test]
    fn test_rnn_forward() {
        let rnn = RNNCell::new(2, 3, Activation::Tanh);
        let input = vec![1.0, 2.0];
        let hidden = vec![0.1, 0.2, 0.3];
        
        let (new_hidden, cache) = rnn.forward(&input, &hidden);
        
        assert_eq!(new_hidden.len(), 3);
        assert_eq!(cache.input, input);
        assert_eq!(cache.hidden, hidden);
    }

    /// Tests RNN backward pass
    #[test]
    fn test_rnn_backward() {
        let rnn = RNNCell::new(2, 3, Activation::Tanh);
        let input = vec![1.0, 2.0];
        let hidden = vec![0.1, 0.2, 0.3];
        
        let (_, cache) = rnn.forward(&input, &hidden);
        let grad_next = vec![0.1, 0.2, 0.3];
        let grad_output = vec![0.1, 0.2, 0.3];
        
        let (grad_input, grad_hidden, gradients) = rnn.backward(&grad_next, &grad_output, &cache);
        
        assert_eq!(grad_input.len(), 2);
        assert_eq!(grad_hidden.len(), 3);
        assert_eq!(gradients.w_ih.len(), 3);
        assert_eq!(gradients.w_hh.len(), 3);
        assert_eq!(gradients.bias.len(), 3);
    }

    /// Tests LSTM cell initialization
    #[test]
    fn test_lstm_cell_initialization() {
        let lstm = LSTMCell::new(3, 4);
        assert_eq!(lstm.input_size, 3);
        assert_eq!(lstm.hidden_size, 4);
        assert_eq!(lstm.w_ii.len(), 4);
        assert_eq!(lstm.w_ii[0].len(), 3);
        assert_eq!(lstm.w_hi.len(), 4);
        assert_eq!(lstm.w_hi[0].len(), 4);
        assert_eq!(lstm.bias.len(), 16); // 4 * hidden_size
    }

    /// Tests LSTM forward pass
    #[test]
    fn test_lstm_forward() {
        let lstm = LSTMCell::new(2, 3);
        let input = vec![1.0, 2.0];
        let hidden = vec![0.1, 0.2, 0.3];
        let cell = vec![0.1, 0.2, 0.3];
        
        let (new_hidden, new_cell, cache) = lstm.forward(&input, &hidden, &cell);
        
        assert_eq!(new_hidden.len(), 3);
        assert_eq!(new_cell.len(), 3);
        assert_eq!(cache.input, input);
        assert_eq!(cache.hidden, hidden);
        assert_eq!(cache.cell, cell);
    }

    /// Tests LSTM backward pass
    #[test]
    fn test_lstm_backward() {
        let lstm = LSTMCell::new(2, 3);
        let input = vec![1.0, 2.0];
        let hidden = vec![0.1, 0.2, 0.3];
        let cell = vec![0.1, 0.2, 0.3];
        
        let (_, _, cache) = lstm.forward(&input, &hidden, &cell);
        let grad_next_h = vec![0.1, 0.2, 0.3];
        let grad_next_c = vec![0.1, 0.2, 0.3];
        
        let (grad_input, grad_hidden, grad_cell, gradients) = 
            lstm.backward(&grad_next_h, &grad_next_c, &cache);
        
        assert_eq!(grad_input.len(), 2);
        assert_eq!(grad_hidden.len(), 3);
        assert_eq!(grad_cell.len(), 3);
        assert_eq!(gradients.w_ii.len(), 3);
        assert_eq!(gradients.w_hi.len(), 3);
        assert_eq!(gradients.bias.len(), 12);
    }

    /// Tests input size validation
    #[test]
    #[should_panic(expected = "Input size mismatch")]
    fn test_rnn_input_validation() {
        let rnn = RNNCell::new(2, 3, Activation::Tanh);
        let input = vec![1.0, 2.0, 3.0]; // Wrong input size
        let hidden = vec![0.1, 0.2, 0.3];
        rnn.forward(&input, &hidden);
    }

    /// Tests hidden size validation
    #[test]
    #[should_panic(expected = "Hidden size mismatch")]
    fn test_lstm_hidden_validation() {
        let lstm = LSTMCell::new(2, 3);
        let input = vec![1.0, 2.0];
        let hidden = vec![0.1, 0.2]; // Wrong hidden size
        let cell = vec![0.1, 0.2, 0.3];
        lstm.forward(&input, &hidden, &cell);
    }
}
