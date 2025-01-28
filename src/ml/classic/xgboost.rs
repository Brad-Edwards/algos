use rand::prelude::*;
use std::f64;

/// The training objective for XGBoost-style gradient boosting.
#[derive(Debug, Clone)]
pub enum XGBObjective {
    /// Regression with squared error loss:
    ///   gradient = (pred - label)
    ///   hessian = 1
    RegSquareError,
    /// Binary logistic classification (labels in {0,1}).
    /// We treat the model output as log-odds.
    /// gradient = p - label, hessian = p*(1-p), where p = sigmoid(pred).
    BinaryLogistic,
}

/// Configuration parameters for an XGBoost-like model.
#[derive(Debug, Clone)]
pub struct XGBConfig {
    /// Number of boosting rounds.
    pub n_estimators: usize,
    /// Maximum tree depth.
    pub max_depth: usize,
    /// L2 regularization on leaf weights (commonly called `lambda`).
    pub lambda: f64,
    /// L1 regularization on leaf weights (commonly called `alpha`).
    pub alpha: f64,
    /// Minimum loss reduction required to make a further partition on a leaf node (gamma).
    pub gamma: f64,
    /// Minimum sum of hessians needed in a leaf (min_child_weight).
    pub min_child_weight: f64,
    /// Subsample ratio of the training instances (row subsampling).
    pub subsample: f64,
    /// Subsample ratio of columns when constructing each tree (colsample_bytree).
    pub colsample_bytree: f64,
    /// Learning rate (eta).
    pub learning_rate: f64,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

/// A minimal XGBoost-inspired gradient boosting model for either regression or binary logistic.
#[derive(Debug)]
pub struct XGBoostModel {
    /// The objective function (determines gradients/hessians).
    pub objective: XGBObjective,
    /// A list of learned trees.
    pub trees: Vec<XGBTree>,
    /// The config used for training.
    pub config: XGBConfig,
    /// Initial prediction (bias). E.g. mean for regression, log( pos/neg ) for logistic, etc.
    pub base_score: f64,
}

/// A single regression tree node storing split structure or a leaf value.
#[derive(Debug, Clone)]
enum XGBTreeNode {
    /// Leaf node with a constant weight (score) contribution.
    Leaf(f64),
    /// Internal node that splits on feature_index <= threshold.
    Internal {
        feature_index: usize,
        threshold: f64,
        left_child: Box<XGBTreeNode>,
        right_child: Box<XGBTreeNode>,
    },
}

/// A CART-style regression tree used by XGBoost, storing node splits with
/// second-order statistics to guide splitting.
#[derive(Debug, Clone)]
pub struct XGBTree {
    root: XGBTreeNode,
}

impl XGBTree {
    /// Predict the contribution of this tree for a single sample.
    pub fn predict_one(&self, sample: &[f64]) -> f64 {
        traverse(&self.root, sample)
    }
}

/// Implementation of XGBoost model
impl XGBoostModel {
    /// Create a new model with given objective and config.
    pub fn new(objective: XGBObjective, config: XGBConfig) -> Self {
        Self {
            objective,
            trees: Vec::new(),
            config,
            base_score: 0.0,
        }
    }

    /// Fit the XGBoost model on the dataset (features, labels).
    ///
    /// For `BinaryLogistic`, labels must be 0.0 or 1.0.
    /// For `RegSquareError`, labels can be any numeric value.
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        let n = x.len();
        if n == 0 {
            panic!("No training data provided to XGBoostModel.");
        }
        if y.len() != n {
            panic!("Features and labels must match in length.");
        }

        // Validate logistic labels
        if let XGBObjective::BinaryLogistic = self.objective {
            for &lbl in y {
                if !(0.0..=1.0).contains(&lbl) {
                    panic!("BinaryLogistic expects labels in [0,1], got {}", lbl);
                }
            }
        }

        // Initialize base_score
        match self.objective {
            XGBObjective::RegSquareError => {
                // Typically mean of y
                self.base_score = mean(y);
            }
            XGBObjective::BinaryLogistic => {
                // Initialize conservatively with a small value
                // This helps prevent extreme initial predictions
                self.base_score = 0.0; // Start at decision boundary
            }
        }

        let mut rng = match self.config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        // Start with predictions = base_score
        let mut preds = vec![self.base_score; n];

        // Clear any existing trees
        self.trees.clear();

        for _round in 0..self.config.n_estimators {
            // 1) compute grad, hess
            let (grad, hess) = match self.objective {
                XGBObjective::RegSquareError => {
                    // gradient = (pred - y), hessian = 1
                    let mut g = vec![0.0; n];
                    let hh = vec![1.0; n];
                    for i in 0..n {
                        g[i] = preds[i] - y[i];
                    }
                    (g, hh)
                }
                XGBObjective::BinaryLogistic => {
                    // p = sigmoid(pred)
                    // gradient = p - label
                    // hessian = p*(1-p)
                    let mut g = vec![0.0; n];
                    let mut hh = vec![0.0; n];
                    for i in 0..n {
                        let p = 1.0 / (1.0 + (-preds[i]).exp());
                        g[i] = p - y[i];
                        hh[i] = p * (1.0 - p);
                    }
                    (g, hh)
                }
            };

            // 2) row and col subsampling
            let (sample_mask, col_mask) = subsample_masks(
                n,
                x[0].len(),
                self.config.subsample,
                self.config.colsample_bytree,
                &mut rng,
            );

            // 3) build a tree
            let tree = build_xgb_tree(x, &grad, &hess, &sample_mask, &col_mask, &self.config, 0);
            // 4) append the tree
            self.trees.push(XGBTree { root: tree.clone() });

            // 5) update preds: F_m = F_{m-1} + eta * Tree
            for i in 0..n {
                if sample_mask[i] {
                    let incr = traverse(&tree, &x[i]) * self.config.learning_rate;
                    preds[i] += incr;
                }
            }
        }
    }

    /// Predict for a single sample.
    /// - For `RegSquareError`, returns raw sum of base_score + trees.
    /// - For `BinaryLogistic`, returns 0.0 or 1.0 (threshold=0.5 on sigmoid).
    pub fn predict_one(&self, sample: &[f64]) -> f64 {
        let score = self.decision_function_one(sample);
        match self.objective {
            XGBObjective::RegSquareError => score,
            XGBObjective::BinaryLogistic => {
                let p = 1.0 / (1.0 + (-score).exp());
                if p >= 0.5 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    /// The raw model output: base_score + sum of tree outputs.
    /// For logistic, interpret as log-odds.
    pub fn decision_function_one(&self, sample: &[f64]) -> f64 {
        let mut sum_val = self.base_score;
        for tree in &self.trees {
            sum_val += tree.predict_one(sample) * self.config.learning_rate;
        }
        sum_val
    }

    /// Predict multiple samples at once.
    pub fn predict_batch(&self, data: &[Vec<f64>]) -> Vec<f64> {
        data.iter().map(|row| self.predict_one(row)).collect()
    }
}

// ---- Internal functions for building XGB trees ----

#[derive(Clone, Debug)]
struct XGBNodeSplit {
    feature_index: usize,
    threshold: f64,
    left_index: Vec<usize>,
    right_index: Vec<usize>,
    // Stats
    gain: f64,
}

struct SplitParams<'a> {
    x: &'a [Vec<f64>],
    grad: &'a [f64],
    hess: &'a [f64],
    indices: &'a [usize],
    feat_idx: usize,
    lambda: f64,
    alpha: f64,
    g_node: f64,
    h_node: f64,
    min_child_weight: f64,
    col_mask: &'a [bool],
}

/// Build a single XGBoost tree node recursively using second-order stats.
fn build_xgb_tree(
    x: &[Vec<f64>],
    grad: &[f64],
    hess: &[f64],
    sample_mask: &[bool],
    col_mask: &[bool],
    config: &XGBConfig,
    depth: usize,
) -> XGBTreeNode {
    // gather sample indices
    let mut indices = Vec::new();
    for (i, &m) in sample_mask.iter().enumerate() {
        if m {
            indices.push(i);
        }
    }

    // If no data or max_depth reached, make leaf
    if indices.is_empty() || depth >= config.max_depth {
        let leaf_val = compute_leaf_weight(grad, hess, &indices, config);
        return XGBTreeNode::Leaf(leaf_val);
    }

    // sum of grad/hess for this node
    let (g_node, h_node) = sum_grad_hess(grad, hess, &indices);

    // If h_node < min_child_weight, output leaf
    if h_node < config.min_child_weight {
        let leaf_val = calc_gamma(g_node, h_node, config.lambda, config.alpha);
        return XGBTreeNode::Leaf(leaf_val);
    }

    // Find best split among columns allowed by col_mask
    let best_split = find_best_xgb_split(SplitParams {
        x,
        grad,
        hess,
        indices: &indices,
        feat_idx: 0,
        lambda: config.lambda,
        alpha: config.alpha,
        g_node,
        h_node,
        min_child_weight: config.min_child_weight,
        col_mask,
    });
    match best_split {
        None => {
            // no valid split => leaf
            let leaf_val = calc_gamma(g_node, h_node, config.lambda, config.alpha);
            XGBTreeNode::Leaf(leaf_val)
        }
        Some(sp) => {
            // Check if the gain is smaller than gamma => no split
            if sp.gain < config.gamma {
                let leaf_val = calc_gamma(g_node, h_node, config.lambda, config.alpha);
                return XGBTreeNode::Leaf(leaf_val);
            }
            // build children
            let mut left_mask = vec![false; sample_mask.len()];
            for &i in sp.left_index.iter() {
                left_mask[i] = true;
            }
            let mut right_mask = vec![false; sample_mask.len()];
            for &i in sp.right_index.iter() {
                right_mask[i] = true;
            }

            let left_child = build_xgb_tree(x, grad, hess, &left_mask, col_mask, config, depth + 1);
            let right_child =
                build_xgb_tree(x, grad, hess, &right_mask, col_mask, config, depth + 1);

            XGBTreeNode::Internal {
                feature_index: sp.feature_index,
                threshold: sp.threshold,
                left_child: Box::new(left_child),
                right_child: Box::new(right_child),
            }
        }
    }
}

/// Find the best split over the allowed columns.
/// We compute approximate gain using G = sum(grad), H = sum(hess) for left/right subsets.
fn find_best_xgb_split(params: SplitParams) -> Option<XGBNodeSplit> {
    let mut best: Option<XGBNodeSplit> = None;
    let base_score = calc_gain(params.g_node, params.h_node, params.lambda, params.alpha);

    // For each feature
    for (feat_idx, &use_col) in params.col_mask.iter().enumerate() {
        if !use_col {
            continue;
        }

        let split_params = SplitParams {
            x: params.x,
            grad: params.grad,
            hess: params.hess,
            indices: params.indices,
            feat_idx,
            lambda: params.lambda,
            alpha: params.alpha,
            g_node: params.g_node,
            h_node: params.h_node,
            min_child_weight: params.min_child_weight,
            col_mask: params.col_mask,
        };

        if let Some(split) =
            find_best_split_for_feature(split_params, base_score, params.min_child_weight)
        {
            if best.is_none() || split.gain > best.as_ref().unwrap().gain {
                best = Some(split);
            }
        }
    }

    best
}

fn find_best_split_for_feature(
    params: SplitParams,
    base_score: f64,
    min_child_weight: f64,
) -> Option<XGBNodeSplit> {
    let mut vals = Vec::with_capacity(params.indices.len());
    for &i in params.indices.iter() {
        vals.push((params.x[i][params.feat_idx], i));
    }
    vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut best_gain = 0.0;
    let mut best_split = None;

    let mut g_left = 0.0;
    let mut h_left = 0.0;
    let mut left_idx = Vec::new();

    for i in 0..vals.len() - 1 {
        let (v, idx) = vals[i];
        g_left += params.grad[idx];
        h_left += params.hess[idx];
        left_idx.push(idx);

        // check if next value is distinct => potential threshold
        let next_val = vals[i + 1].0;
        if (v - next_val).abs() > f64::EPSILON {
            // evaluate gain
            let g_right = params.g_node - g_left;
            let h_right = params.h_node - h_left;

            if h_left >= min_child_weight && h_right >= min_child_weight {
                let gain_left = calc_gain(g_left, h_left, params.lambda, params.alpha);
                let gain_right = calc_gain(g_right, h_right, params.lambda, params.alpha);
                let gain = gain_left + gain_right - base_score;

                if gain > best_gain {
                    best_gain = gain;
                    let right_idx: Vec<usize> =
                        vals[(i + 1)..].iter().map(|(_, idx)| *idx).collect();
                    best_split = Some(XGBNodeSplit {
                        feature_index: params.feat_idx,
                        threshold: (v + next_val) / 2.0,
                        left_index: left_idx.clone(),
                        right_index: right_idx,
                        gain,
                    });
                }
            }
        }
    }

    best_split
}

/// Subsample a fraction of rows and columns for the tree.
fn subsample_masks(
    n_rows: usize,
    n_cols: usize,
    subsample_ratio: f64,
    colsample_ratio: f64,
    rng: &mut impl Rng,
) -> (Vec<bool>, Vec<bool>) {
    // row sampling
    let mut row_mask = vec![false; n_rows];
    let sample_size = (subsample_ratio * n_rows as f64).ceil() as usize;
    if sample_size >= n_rows {
        // use all
        for mask in row_mask.iter_mut().take(n_rows) {
            *mask = true;
        }
    } else {
        // pick randomly
        let mut indices: Vec<usize> = (0..n_rows).collect();
        indices.shuffle(rng);
        for i in 0..sample_size.min(n_rows) {
            row_mask[indices[i]] = true;
        }
    }

    // column sampling
    let mut col_mask = vec![false; n_cols];
    let col_sample_size = (colsample_ratio * n_cols as f64).ceil() as usize;
    if col_sample_size >= n_cols {
        // use all
        for mask in col_mask.iter_mut().take(n_cols) {
            *mask = true;
        }
    } else {
        let mut indices: Vec<usize> = (0..n_cols).collect();
        indices.shuffle(rng);
        for i in 0..col_sample_size.min(n_cols) {
            col_mask[indices[i]] = true;
        }
    }
    (row_mask, col_mask)
}

/// Summation of gradient/hessian over the specified subset indices.
fn sum_grad_hess(grad: &[f64], hess: &[f64], indices: &[usize]) -> (f64, f64) {
    let mut g = 0.0;
    let mut hh = 0.0;
    for &i in indices {
        g += grad[i];
        hh += hess[i];
    }
    (g, hh)
}

/// Leaf weight formula for XGBoost:
///   w* = -G / (H + lambda), ignoring gamma here because that's for the splitting gain threshold.
fn compute_leaf_weight(grad: &[f64], hess: &[f64], indices: &[usize], cfg: &XGBConfig) -> f64 {
    let (g, hh) = sum_grad_hess(grad, hess, indices);
    calc_gamma(g, hh, cfg.lambda, cfg.alpha)
}

/// The final leaf weight in presence of L1 (alpha) in XGBoost can be solved by "soft thresholding".
/// In a simplified approach:
///   w* = - sign(G) * max(0, |G| - alpha) / (H + lambda)
fn calc_gamma(g: f64, h: f64, lambda: f64, alpha: f64) -> f64 {
    if h.abs() < f64::EPSILON {
        return 0.0;
    }
    let sign_g = if g > 0.0 { 1.0 } else { -1.0 };
    let abs_g = g.abs();
    let res = (abs_g - alpha).max(0.0) / (h + lambda);
    -sign_g * res
}

/// The gain from splitting one node into two, ignoring gamma here, is:
/// gain = 1/2 * [G_L^2/(H_L + lambda) + G_R^2/(H_R + lambda) - G^2/(H + lambda)]
fn calc_gain(g: f64, h: f64, lambda: f64, _alpha: f64) -> f64 {
    // ignoring alpha for gain formula except in leaf weight.
    // This is a simplified approach.
    // Typically alpha affects the leaf weight, thus the net gain is also influenced indirectly.
    // We'll do standard formula ignoring alpha's direct role in splitting gain.
    if h.abs() < f64::EPSILON {
        return 0.0;
    }
    0.5 * ((g * g) / (h + lambda))
}

/// Tree traversal to get leaf value.
fn traverse(node: &XGBTreeNode, sample: &[f64]) -> f64 {
    match node {
        XGBTreeNode::Leaf(w) => *w,
        XGBTreeNode::Internal {
            feature_index,
            threshold,
            left_child,
            right_child,
        } => {
            if sample[*feature_index] <= *threshold {
                traverse(left_child, sample)
            } else {
                traverse(right_child, sample)
            }
        }
    }
}

/// Helper: Mean of a slice.
fn mean(arr: &[f64]) -> f64 {
    if arr.is_empty() {
        0.0
    } else {
        arr.iter().sum::<f64>() / (arr.len() as f64)
    }
}

// ---- Tests ----
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xgb_reg_square_error() {
        // Simple function y = x1 + 2*x2
        let x = vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 1.0],
            vec![0.0, 0.0],
            vec![4.0, 2.0],
        ];
        let y: Vec<f64> = x.iter().map(|row| row[0] + 2.0 * row[1]).collect();

        let config = XGBConfig {
            n_estimators: 100,
            max_depth: 3,
            lambda: 1.0,
            alpha: 0.0,
            gamma: 0.0,
            min_child_weight: 0.1,
            subsample: 1.0,
            colsample_bytree: 1.0,
            learning_rate: 0.3,
            seed: Some(42),
        };
        let mut model = XGBoostModel::new(XGBObjective::RegSquareError, config);
        model.fit(&x, &y);

        for i in 0..x.len() {
            let pred = model.predict_one(&x[i]);
            let err = (pred - y[i]).abs();
            assert!(err < 1.0, "Prediction error is too large: err={}", err);
        }
    }

    #[test]
    fn test_xgb_binary_logistic() {
        // We'll define a simple classification: label=1 if x1 + x2>3, else 0
        let x = vec![
            vec![0.0, 0.0], // sum=0 => clearly 0
            vec![5.0, 5.0], // sum=10 => clearly 1
            vec![0.0, 1.0], // sum=1 => clearly 0
            vec![5.0, 4.0], // sum=9 => clearly 1
            vec![0.5, 0.5], // sum=1 => clearly 0
        ];
        let y: Vec<f64> = x
            .iter()
            .map(|row| if row[0] + row[1] > 3.0 { 1.0 } else { 0.0 })
            .collect();

        let config = XGBConfig {
            n_estimators: 100,
            max_depth: 3,
            lambda: 1.0,
            alpha: 0.0,
            gamma: 0.0,
            min_child_weight: 0.1,
            subsample: 1.0,
            colsample_bytree: 1.0,
            learning_rate: 0.3,
            seed: Some(123),
        };
        let mut model = XGBoostModel::new(XGBObjective::BinaryLogistic, config);
        model.fit(&x, &y);

        // Check predictions
        for i in 0..x.len() {
            let pred = model.predict_one(&x[i]);
            let truth = y[i];
            let is_correct = (pred - truth).abs() < 0.5;
            assert!(
                is_correct,
                "Wrong classification for row {} => pred={}",
                i, pred
            );
        }
    }
}
