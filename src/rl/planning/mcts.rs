use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// AlphaZero-style Monte Carlo Tree Search.
///
/// Combines a dual-head neural network (policy + value) with MCTS.
/// The network provides prior probabilities and value estimates that
/// guide tree expansion and node evaluation. Actions are selected by
/// highest visit count after simulation.
pub struct AlphaZeroMCTS {
    network: DualNetwork,
    c_puct: f64,
    n_simulations: usize,
    root: Option<Node>,
}

impl AlphaZeroMCTS {
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_dim: usize,
        c_puct: f64,
        n_simulations: usize,
    ) -> Self {
        Self {
            network: DualNetwork::new(state_dim, action_dim, hidden_dim),
            c_puct,
            n_simulations,
            root: None,
        }
    }

    /// Run MCTS from the given state and return the best action.
    pub fn search(&mut self, state: Array1<f64>) -> usize {
        let (policy, value) = self.network.predict(&state);
        self.root = Some(Node::new(state, policy, value));

        for _ in 0..self.n_simulations {
            let node = self.root.as_mut().unwrap();
            let mut path = Vec::new();
            let mut current = node as *mut Node;

            // Selection: descend to a leaf
            unsafe {
                while !(*current).is_terminal && (*current).is_fully_expanded() {
                    let action = (*current).select_action(self.c_puct);
                    path.push(action);
                    current = (*current).children.get_mut(&action).unwrap() as *mut Node;
                }

                // Expansion and evaluation
                if !(*current).is_terminal {
                    let (policy, value) = self.network.predict(&(*current).state);
                    (*current).expand(policy);
                    (*current).backup_value(value, &path);
                }
            }
        }

        self.root.as_ref().unwrap().best_action_by_visits()
    }

    /// Update tree with observed transition (simplified self-play update).
    pub fn update(&mut self, action: usize, reward: f64, next_state: &Array1<f64>, done: bool) {
        if let Some(root) = self.root.as_mut() {
            if let Some(child) = root.children.get_mut(&action) {
                child.value_sum[action] += reward;
                child.visit_count[action] += 1;
                if done {
                    child.is_terminal = true;
                } else {
                    let (policy, _) = self.network.predict(next_state);
                    child.prior_p = policy;
                }
            }
        }
    }
}

struct Node {
    state: Array1<f64>,
    prior_p: Array1<f64>,
    visit_count: Array1<usize>,
    value_sum: Array1<f64>,
    children: HashMap<usize, Node>,
    is_terminal: bool,
}

impl Node {
    fn new(state: Array1<f64>, prior_p: Array1<f64>, _value: f64) -> Self {
        let n = prior_p.len();
        Self {
            state,
            prior_p,
            visit_count: Array1::zeros(n),
            value_sum: Array1::zeros(n),
            children: HashMap::new(),
            is_terminal: false,
        }
    }

    fn select_action(&self, c_puct: f64) -> usize {
        let total: f64 = self.visit_count.sum() as f64 + 1e-8;
        (0..self.prior_p.len())
            .max_by(|&a, &b| {
                let score = |act: usize| {
                    let q = if self.visit_count[act] > 0 {
                        self.value_sum[act] / self.visit_count[act] as f64
                    } else {
                        0.0
                    };
                    q + c_puct
                        * self.prior_p[act]
                        * (total.sqrt() / (1.0 + self.visit_count[act] as f64))
                };
                score(a).partial_cmp(&score(b)).unwrap()
            })
            .unwrap_or(0)
    }

    fn best_action_by_visits(&self) -> usize {
        self.visit_count
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn expand(&mut self, policy: Array1<f64>) {
        for (action, &p) in policy.iter().enumerate() {
            if p > 0.0 && !self.children.contains_key(&action) {
                let next_state = self.state.clone();
                self.children
                    .insert(action, Node::new(next_state, policy.clone(), 0.0));
            }
        }
    }

    fn backup_value(&mut self, value: f64, actions: &[usize]) {
        for &a in actions {
            self.value_sum[a] += value;
            self.visit_count[a] += 1;
        }
    }

    fn is_fully_expanded(&self) -> bool {
        self.children.len() == self.prior_p.iter().filter(|&&p| p > 0.0).count()
    }
}

struct DualNetwork {
    shared: SharedLayers,
    policy_head: PolicyHead,
    value_head: ValueHead,
}

impl DualNetwork {
    fn new(state_dim: usize, action_dim: usize, hidden_dim: usize) -> Self {
        Self {
            shared: SharedLayers::new(state_dim, hidden_dim),
            policy_head: PolicyHead::new(hidden_dim, action_dim),
            value_head: ValueHead::new(hidden_dim),
        }
    }

    fn predict(&self, state: &Array1<f64>) -> (Array1<f64>, f64) {
        let hidden = self.shared.forward(state);
        (
            self.policy_head.forward(&hidden),
            self.value_head.forward(&hidden),
        )
    }
}

struct SharedLayers {
    w1: Array2<f64>,
    b1: Array1<f64>,
    w2: Array2<f64>,
    b2: Array1<f64>,
}

impl SharedLayers {
    fn new(input_dim: usize, hidden_dim: usize) -> Self {
        Self {
            w1: Array2::zeros((input_dim, hidden_dim)),
            b1: Array1::zeros(hidden_dim),
            w2: Array2::zeros((hidden_dim, hidden_dim)),
            b2: Array1::zeros(hidden_dim),
        }
    }

    fn forward(&self, state: &Array1<f64>) -> Array1<f64> {
        let h1 = (state.dot(&self.w1) + &self.b1).mapv(|x| x.max(0.0));
        (h1.dot(&self.w2) + &self.b2).mapv(|x| x.max(0.0))
    }
}

struct PolicyHead {
    weights: Array2<f64>,
    biases: Array1<f64>,
}

impl PolicyHead {
    fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            weights: Array2::zeros((input_dim, output_dim)),
            biases: Array1::zeros(output_dim),
        }
    }

    fn forward(&self, hidden: &Array1<f64>) -> Array1<f64> {
        let logits = hidden.dot(&self.weights) + &self.biases;
        let max_val = logits.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp = logits.mapv(|x| (x - max_val).exp());
        exp.clone() / exp.sum()
    }
}

struct ValueHead {
    weights: Array2<f64>,
    biases: Array1<f64>,
}

impl ValueHead {
    fn new(input_dim: usize) -> Self {
        Self {
            weights: Array2::zeros((input_dim, 1)),
            biases: Array1::zeros(1),
        }
    }

    fn forward(&self, hidden: &Array1<f64>) -> f64 {
        (hidden.dot(&self.weights) + &self.biases)[0].tanh()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_select_action_deterministic() {
        let priors = Array1::from(vec![0.5, 0.3, 0.2]);
        let node = Node::new(Array1::zeros(4), priors, 0.0);
        // With no visits, highest prior should win
        let action = node.select_action(1.0);
        assert_eq!(action, 0);
    }

    #[test]
    fn test_dual_network_output_shapes() {
        let net = DualNetwork::new(4, 3, 8);
        let state = Array1::zeros(4);
        let (policy, value) = net.predict(&state);
        assert_eq!(policy.len(), 3);
        assert!((policy.sum() - 1.0).abs() < 1e-10);
        assert!(value >= -1.0 && value <= 1.0);
    }
}
