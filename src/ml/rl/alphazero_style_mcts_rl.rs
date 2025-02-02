use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// AlphaZero-style Monte Carlo Tree Search with RL implementation
pub struct AlphaZeroStyleMCTS {
    network: Network,
    c_puct: f64,
    n_simulations: usize,
    root: Option<Node>,
}

struct Network {
    shared_layers: SharedLayers,
    policy_head: PolicyHead,
    value_head: ValueHead,
}

struct SharedLayers {
    weights1: Array2<f64>,
    biases1: Array1<f64>,
    weights2: Array2<f64>,
    biases2: Array1<f64>,
}

struct PolicyHead {
    weights: Array2<f64>,
    biases: Array1<f64>,
}

struct ValueHead {
    weights: Array2<f64>,
    biases: Array1<f64>,
}

struct Node {
    state: Array1<f64>,
    prior_p: Array1<f64>,
    visit_count: Array1<usize>,
    value_sum: Array1<f64>,
    children: HashMap<usize, Node>,
    is_terminal: bool,
}

impl AlphaZeroStyleMCTS {
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_dim: usize,
        c_puct: f64,
        n_simulations: usize,
    ) -> Self {
        AlphaZeroStyleMCTS {
            network: Network::new(state_dim, action_dim, hidden_dim),
            c_puct,
            n_simulations,
            root: None,
        }
    }

    pub fn search(&mut self, state: Array1<f64>) -> usize {
        // Initialize root node
        let (policy, value) = self.network.predict(&state);
        self.root = Some(Node::new(state, policy, value));

        // Perform MCTS simulations
        for _ in 0..self.n_simulations {
            let mut node = self.root.as_mut().unwrap();
            let mut path = Vec::new();

            // Selection
            while !node.is_terminal && node.is_fully_expanded() {
                let action = node.select_action(self.c_puct);
                path.push(action);
                node = node.children.get_mut(&action).unwrap();
            }

            // Expansion and evaluation
            if !node.is_terminal {
                let (policy, value) = self.network.predict(&node.state);
                node.expand(policy);
                node.backup_value(value, &path);
            }
        }

        // Select action with highest visit count
        self.root.as_ref().unwrap().select_best_action()
    }

    pub fn update_with_self_play(
        &mut self,
        _state: Array1<f64>,
        action: usize,
        reward: f64,
        next_state: Array1<f64>,
        done: bool,
    ) {
        // Store experience for training (implementation would depend on training strategy)
        // This is a simplified version - real implementation would need more sophisticated
        // training logic similar to the original AlphaZero paper
        if let Some(node) = self.root.as_mut() {
            if let Some(child) = node.children.get_mut(&action) {
                child.value_sum[action] += reward;
                child.visit_count[action] += 1;

                if done {
                    child.is_terminal = true;
                } else {
                    let (policy, _) = self.network.predict(&next_state);
                    child.prior_p = policy;
                }
            }
        }
    }
}

impl Network {
    fn new(state_dim: usize, action_dim: usize, hidden_dim: usize) -> Self {
        Network {
            shared_layers: SharedLayers::new(state_dim, hidden_dim),
            policy_head: PolicyHead::new(hidden_dim, action_dim),
            value_head: ValueHead::new(hidden_dim),
        }
    }

    fn predict(&self, state: &Array1<f64>) -> (Array1<f64>, f64) {
        let hidden = self.shared_layers.forward(state);
        let policy = self.policy_head.forward(&hidden);
        let value = self.value_head.forward(&hidden);

        (policy, value)
    }
}

impl SharedLayers {
    fn new(input_dim: usize, hidden_dim: usize) -> Self {
        SharedLayers {
            weights1: Array2::zeros((input_dim, hidden_dim)),
            biases1: Array1::zeros(hidden_dim),
            weights2: Array2::zeros((hidden_dim, hidden_dim)),
            biases2: Array1::zeros(hidden_dim),
        }
    }

    fn forward(&self, state: &Array1<f64>) -> Array1<f64> {
        let hidden1 = state.dot(&self.weights1) + &self.biases1;
        let hidden1 = hidden1.mapv(|x| x.max(0.0)); // ReLU

        let hidden2 = hidden1.dot(&self.weights2) + &self.biases2;
        hidden2.mapv(|x| x.max(0.0)) // ReLU
    }
}

impl PolicyHead {
    fn new(input_dim: usize, output_dim: usize) -> Self {
        PolicyHead {
            weights: Array2::zeros((input_dim, output_dim)),
            biases: Array1::zeros(output_dim),
        }
    }

    fn forward(&self, hidden: &Array1<f64>) -> Array1<f64> {
        let logits = hidden.dot(&self.weights) + &self.biases;
        self.softmax(logits)
    }

    fn softmax(&self, x: Array1<f64>) -> Array1<f64> {
        let max_val = x.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_x = x.mapv(|a| (a - max_val).exp());
        let sum_exp = exp_x.sum();
        exp_x / sum_exp
    }
}

impl ValueHead {
    fn new(input_dim: usize) -> Self {
        ValueHead {
            weights: Array2::zeros((input_dim, 1)),
            biases: Array1::zeros(1),
        }
    }

    fn forward(&self, hidden: &Array1<f64>) -> f64 {
        let value = hidden.dot(&self.weights) + &self.biases;
        value[0].tanh() // Output in [-1, 1]
    }
}

impl Node {
    fn new(state: Array1<f64>, prior_p: Array1<f64>, _value: f64) -> Self {
        let action_dim = prior_p.len();
        Node {
            state,
            prior_p,
            visit_count: Array1::zeros(action_dim),
            value_sum: Array1::zeros(action_dim),
            children: HashMap::new(),
            is_terminal: false,
        }
    }

    fn select_action(&self, c_puct: f64) -> usize {
        let total_count: f64 = self.visit_count.sum() as f64 + 1e-8;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_action = 0;

        for (action, &prior) in self.prior_p.iter().enumerate() {
            let q_value = if self.visit_count[action] > 0 {
                self.value_sum[action] / self.visit_count[action] as f64
            } else {
                0.0
            };

            let u_value =
                c_puct * prior * (total_count.sqrt() / (1.0 + self.visit_count[action] as f64));
            let score = q_value + u_value;

            if score > best_score {
                best_score = score;
                best_action = action;
            }
        }

        best_action
    }

    fn select_best_action(&self) -> usize {
        let mut best_count = 0;
        let mut best_action = 0;

        for (action, &count) in self.visit_count.iter().enumerate() {
            if count > best_count {
                best_count = count;
                best_action = action;
            }
        }

        best_action
    }

    fn expand(&mut self, policy: Array1<f64>) {
        for (action, &p) in policy.iter().enumerate() {
            if p > 0.0 && !self.children.contains_key(&action) {
                let next_state = self.state.clone();
                // State transition would happen here in real implementation
                self.children
                    .insert(action, Node::new(next_state, policy.clone(), 0.0));
            }
        }
    }

    fn backup_value(&mut self, value: f64, actions: &[usize]) {
        for &action in actions.iter() {
            self.value_sum[action] += value;
            self.visit_count[action] += 1;
        }
    }

    fn is_fully_expanded(&self) -> bool {
        self.children.len() == self.prior_p.iter().filter(|&&p| p > 0.0).count()
    }
}
