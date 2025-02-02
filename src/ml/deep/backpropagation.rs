use ndarray::{Array1, Array2, Axis};
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

pub trait Layer {
    /// Forward pass: takes input (batch_size, in_features).
    fn forward(&mut self, input: &Array2<f64>) -> Array2<f64>;
    /// Backward pass: given grad w.r.t. layer output, returns grad w.r.t. layer input.
    fn backward(&mut self, grad_output: &Array2<f64>) -> Array2<f64>;
    /// Update internal parameters using stored gradients.
    fn update_params(&mut self, learning_rate: f64);
}

/// A dense (fully connected) layer with weights + biases.
pub struct DenseLayer {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,

    input_cache: Option<Array2<f64>>,
    weight_grads: Option<Array2<f64>>,
    bias_grads: Option<Array1<f64>>,
}

impl DenseLayer {
    pub fn new(in_features: usize, out_features: usize, init_std: f64) -> Self {
        let mut rng = thread_rng();
        let dist = Normal::new(0.0, init_std).unwrap();

        let weights = Array2::from_shape_fn((in_features, out_features), |_| dist.sample(&mut rng));
        let biases = Array1::zeros(out_features);

        Self {
            weights,
            biases,
            input_cache: None,
            weight_grads: None,
            bias_grads: None,
        }
    }
}

#[allow(non_snake_case)]
impl Layer for DenseLayer {
    fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        self.input_cache = Some(input.clone());
        let mut output = input.dot(&self.weights);
        output += &self.biases;
        output
    }

    fn backward(&mut self, grad_output: &Array2<f64>) -> Array2<f64> {
        let input = self
            .input_cache
            .as_ref()
            .expect("Must call forward before backward.");

        // dW = input^T * grad_output
        let dW = input.t().dot(grad_output);
        // dB = sum of grad_output over the batch
        let dB = grad_output.sum_axis(Axis(0));
        // dX = grad_output * W^T
        let dX = grad_output.dot(&self.weights.t());

        self.weight_grads = Some(dW);
        self.bias_grads = Some(dB);

        dX
    }

    fn update_params(&mut self, lr: f64) {
        if let Some(dw) = &self.weight_grads {
            self.weights = &self.weights - &(dw * lr);
        }
        if let Some(db) = &self.bias_grads {
            self.biases = &self.biases - &(db * lr);
        }
        self.input_cache = None;
        self.weight_grads = None;
        self.bias_grads = None;
    }
}

/// Simple sigmoid activation layer
pub struct Sigmoid {
    output_cache: Option<Array2<f64>>,
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Sigmoid {
    pub fn new() -> Self {
        Self { output_cache: None }
    }
}

impl Layer for Sigmoid {
    fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        let output = input.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        self.output_cache = Some(output.clone());
        output
    }

    fn backward(&mut self, grad_output: &Array2<f64>) -> Array2<f64> {
        let out = self.output_cache.as_ref().unwrap();
        // dSigmoid = sigmoid(x) * (1 - sigmoid(x))

        out * (1.0 - out) * grad_output
    }

    fn update_params(&mut self, _lr: f64) {
        self.output_cache = None;
    }
}

/// A small sequential network with multiple layers.
pub struct SequentialNN {
    pub layers: Vec<Box<dyn Layer>>,
    pub learning_rate: f64,
}

impl SequentialNN {
    pub fn new(layers: Vec<Box<dyn Layer>>, learning_rate: f64) -> Self {
        Self {
            layers,
            learning_rate,
        }
    }

    /// Forward pass through the entire network
    pub fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        let mut x = input.clone();
        for layer in self.layers.iter_mut() {
            x = layer.forward(&x);
        }
        x
    }

    /// Backward pass
    pub fn backward(&mut self, grad_output: &Array2<f64>) {
        let mut grad = grad_output.clone();
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad);
        }
    }

    /// Update all params
    pub fn update_params(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.update_params(self.learning_rate);
        }
    }

    /// Mean-squared-error
    pub fn mse_loss(&mut self, inputs: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        let preds = self.forward(inputs);
        let diff = &preds - targets;
        diff.mapv(|x| x.powi(2)).mean().unwrap_or(0.0)
    }
}

/// Train for one epoch using **stochastic gradient descent** (mini-batch style).
/// - `batch_size = 1` is "pure" SGD.
/// - Larger batch_size is "mini-batch" SGD.
pub fn train_sgd(
    net: &mut SequentialNN,
    inputs: &Array2<f64>,
    targets: &Array2<f64>,
    batch_size: usize,
) {
    let n_samples = inputs.len_of(Axis(0));
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(&mut thread_rng());

    // Loop over mini-batches in random order
    for chunk in indices.chunks(batch_size) {
        // Gather the current mini-batch
        let batch_input = Array2::from_shape_fn((chunk.len(), inputs.len_of(Axis(1))), |(i, j)| {
            inputs[[chunk[i], j]]
        });
        let batch_target =
            Array2::from_shape_fn((chunk.len(), targets.len_of(Axis(1))), |(i, j)| {
                targets[[chunk[i], j]]
            });

        // Forward
        let preds = net.forward(&batch_input);
        // MSE derivative: d(0.5*MSE)/dpreds = (preds - batch_target)
        let grad_loss = &preds - &batch_target;

        // Backward
        net.backward(&grad_loss);

        // Update
        net.update_params();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_dense_layer_forward() {
        let mut layer = DenseLayer::new(2, 3, 0.1);
        // Set deterministic weights and biases for testing
        layer.weights = Array2::from_shape_vec((2, 3), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
        layer.biases = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        let input = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let output = layer.forward(&input);

        // Manual calculation: input · weights + biases
        assert_relative_eq!(output[[0, 0]], 1.0 * 0.1 + 2.0 * 0.4 + 0.1, epsilon = 1e-10);
        assert_relative_eq!(output[[0, 1]], 1.0 * 0.2 + 2.0 * 0.5 + 0.2, epsilon = 1e-10);
        assert_relative_eq!(output[[0, 2]], 1.0 * 0.3 + 2.0 * 0.6 + 0.3, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid_activation() {
        let mut sigmoid = Sigmoid::new();
        let input = Array2::from_shape_vec((1, 3), vec![0.0, 1.0, -1.0]).unwrap();
        let output = sigmoid.forward(&input);

        // Test sigmoid(0) = 0.5
        assert_relative_eq!(output[[0, 0]], 0.5, epsilon = 1e-10);
        // Test sigmoid(1) ≈ 0.731...
        assert_relative_eq!(
            output[[0, 1]],
            1.0 / (1.0 + (-1.0f64).exp()),
            epsilon = 1e-10
        );
        // Test sigmoid(-1) ≈ 0.269...
        assert_relative_eq!(output[[0, 2]], 1.0 / (1.0 + 1.0f64.exp()), epsilon = 1e-10);
    }

    #[test]
    fn test_dense_layer_backward() {
        let mut layer = DenseLayer::new(2, 2, 0.1);
        layer.weights = Array2::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        layer.biases = Array1::from_vec(vec![0.1, 0.2]);

        let input = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        layer.forward(&input);

        let grad_output = Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap();
        let grad_input = layer.backward(&grad_output);

        // Check gradient shapes
        assert_eq!(grad_input.shape(), &[1, 2]);
        assert!(layer.weight_grads.is_some());
        assert!(layer.bias_grads.is_some());
    }

    #[test]
    fn test_sequential_network() {
        let mut net = SequentialNN::new(
            vec![
                Box::new(DenseLayer::new(2, 3, 0.1)),
                Box::new(Sigmoid::new()),
                Box::new(DenseLayer::new(3, 1, 0.1)),
            ],
            0.1,
        );

        // Test forward pass
        let input = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let output = net.forward(&input);
        assert_eq!(output.shape(), &[1, 1]);

        // Test loss calculation
        let target = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let loss = net.mse_loss(&input, &target);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_sgd_training() {
        let mut net = SequentialNN::new(
            vec![
                Box::new(DenseLayer::new(2, 3, 0.1)),
                Box::new(Sigmoid::new()),
                Box::new(DenseLayer::new(3, 1, 0.1)),
            ],
            0.1,
        );

        // XOR problem inputs and outputs
        let inputs =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).unwrap();

        let targets = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 1.0, 0.0]).unwrap();

        // Initial loss
        let initial_loss = net.mse_loss(&inputs, &targets);

        // Train for a few epochs
        for _ in 0..100 {
            train_sgd(&mut net, &inputs, &targets, 2);
        }

        // Final loss should be lower than initial loss
        let final_loss = net.mse_loss(&inputs, &targets);
        assert!(final_loss < initial_loss, "Training should reduce loss");
    }
}
