/// Enum specifying the objective function for gradient boosting.
#[derive(Debug, Clone)]
pub enum GBMObjective {
    /// Mean Squared Error for regression.
    MSE,
    /// Binary Logistic: labels must be 0.0 or 1.0; uses logistic loss.
    BinaryLogistic,
}

/// Configuration for the Gradient Boosting model.
#[derive(Debug, Clone)]
pub struct GBMConfig {
    /// Number of trees to fit.
    pub n_estimators: usize,
    /// Learning rate (shrinkage factor).
    pub learning_rate: f64,
    /// Maximum depth of each tree.
    pub max_depth: usize,
    /// Minimum number of samples required to split an internal node.
    pub min_samples_split: usize,
    /// Random seed for reproducibility (optional).
    pub seed: Option<u64>,
}

/// A simple Gradient Boosting model that uses small CART-like regression trees as weak learners.
///
/// Supports:
/// - `GBMObjective::MSE` for regression.
/// - `GBMObjective::BinaryLogistic` for binary classification (labels 0.0 or 1.0).
#[derive(Debug)]
pub struct GradientBoostedModel {
    /// The ensemble of weak learners (regression trees).
    pub trees: Vec<DecisionTreeRegressor>,
    /// One tree for each boosting iteration.
    pub objective: GBMObjective,
    /// Model config for reference.
    pub config: GBMConfig,
    /// Initial prediction (for MSE, often the mean target; for logistic, e.g. log odds).
    pub init_pred: f64,
}

/// A minimal regression tree node for CART-like splitting.
#[derive(Debug, Clone)]
enum TreeNode {
    /// A leaf node predicting a constant value.
    Leaf(f64),
    /// An internal node that splits on a feature index with threshold,
    /// storing left and right child nodes.
    Internal {
        feature_index: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

/// A small regression tree for fitting residuals or pseudo-residuals in GBM.
#[derive(Debug, Clone)]
pub struct DecisionTreeRegressor {
    root: TreeNode,
    max_depth: usize,
    min_samples_split: usize,
}

impl DecisionTreeRegressor {
    /// Create a new uninitialized DecisionTreeRegressor with the given limits.
    pub fn new(max_depth: usize, min_samples_split: usize) -> Self {
        Self {
            root: TreeNode::Leaf(0.0),
            max_depth,
            min_samples_split,
        }
    }

    /// Fit the tree on features X and target y (usually residuals).
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        assert!(!x.is_empty(), "No data for tree fitting.");
        assert_eq!(x.len(), y.len(), "X and y length mismatch.");
        let root_node = build_tree_recursive(x, y, self.max_depth, self.min_samples_split, 0);
        self.root = root_node;
    }

    /// Predict a single sample using the fitted tree.
    pub fn predict_one(&self, sample: &[f64]) -> f64 {
        traverse(&self.root, sample)
    }

    /// Predict multiple samples at once.
    pub fn predict_batch(&self, data: &[Vec<f64>]) -> Vec<f64> {
        data.iter().map(|row| self.predict_one(row)).collect()
    }
}

/// Recursively build the CART-like regression tree.
fn build_tree_recursive(
    x: &[Vec<f64>],
    y: &[f64],
    max_depth: usize,
    min_samples_split: usize,
    current_depth: usize,
) -> TreeNode {
    // Stopping conditions
    if current_depth >= max_depth || x.len() < min_samples_split || is_constant(y) {
        return TreeNode::Leaf(mean(y));
    }

    // Find the best split among all features
    let (best_feat, best_threshold, best_loss, left_idx, right_idx) = find_best_split(x, y);
    if left_idx.is_empty() || right_idx.is_empty() || best_loss < 1e-15 {
        // no effective split
        return TreeNode::Leaf(mean(y));
    }

    let left_x: Vec<Vec<f64>> = left_idx.iter().map(|&i| x[i].clone()).collect();
    let left_y: Vec<f64> = left_idx.iter().map(|&i| y[i]).collect();

    let right_x: Vec<Vec<f64>> = right_idx.iter().map(|&i| x[i].clone()).collect();
    let right_y: Vec<f64> = right_idx.iter().map(|&i| y[i]).collect();

    let left_child = build_tree_recursive(
        &left_x,
        &left_y,
        max_depth,
        min_samples_split,
        current_depth + 1,
    );
    let right_child = build_tree_recursive(
        &right_x,
        &right_y,
        max_depth,
        min_samples_split,
        current_depth + 1,
    );

    TreeNode::Internal {
        feature_index: best_feat,
        threshold: best_threshold,
        left: Box::new(left_child),
        right: Box::new(right_child),
    }
}

/// Find best split by scanning all features for the minimum MSE split.
fn find_best_split(x: &[Vec<f64>], y: &[f64]) -> (usize, f64, f64, Vec<usize>, Vec<usize>) {
    let n = x.len();
    let d = x[0].len();

    let base_loss = variance(y) * (n as f64); // total sum of squares

    let mut best_feat = 0;
    let mut best_threshold = 0.0;
    let mut best_loss = 0.0; // measure improvement in "sum of squares"
    let mut best_left = Vec::new();
    let mut best_right = Vec::new();

    let mut best_reduction = 0.0;

    for feat_idx in 0..d {
        // Gather values
        let mut values: Vec<(f64, usize)> = x
            .iter()
            .enumerate()
            .map(|(i, row)| (row[feat_idx], i))
            .collect();
        // Sort
        values.sort_by(|(v1, _), (v2, _)| v1.partial_cmp(v2).unwrap());

        // We'll try midpoints between distinct sorted values
        for w in values.windows(2) {
            let (val1, _idx1) = w[0];
            let (val2, _idx2) = w[1];
            if (val1 - val2).abs() < 1e-15 {
                continue;
            }
            let threshold = 0.5 * (val1 + val2);

            // Partition
            let mut left_idx = Vec::new();
            let mut right_idx = Vec::new();
            for (v, irow) in &values {
                if *v <= threshold {
                    left_idx.push(*irow);
                } else {
                    right_idx.push(*irow);
                }
            }

            if left_idx.is_empty() || right_idx.is_empty() {
                continue;
            }

            // Weighted sum of variances
            let lvar = variance_subset(y, &left_idx);
            let rvar = variance_subset(y, &right_idx);
            let nl = left_idx.len() as f64;
            let nr = right_idx.len() as f64;
            let split_loss = nl * lvar + nr * rvar; // sum of squares after split

            let reduction = base_loss - split_loss;
            if reduction > best_reduction {
                best_reduction = reduction;
                best_feat = feat_idx;
                best_threshold = threshold;
                best_loss = split_loss;
                best_left = left_idx.clone();
                best_right = right_idx.clone();
            }
        }
    }

    (best_feat, best_threshold, best_loss, best_left, best_right)
}

/// Evaluate a sample by traversing the tree.
fn traverse(node: &TreeNode, row: &[f64]) -> f64 {
    match node {
        TreeNode::Leaf(value) => *value,
        TreeNode::Internal {
            feature_index,
            threshold,
            left,
            right,
        } => {
            if row[*feature_index] <= *threshold {
                traverse(left, row)
            } else {
                traverse(right, row)
            }
        }
    }
}

/// Check if all y values are (nearly) identical.
fn is_constant(y: &[f64]) -> bool {
    if y.is_empty() {
        return true;
    }
    let first = y[0];
    y.iter().all(|val| (val - first).abs() < 1e-15)
}

/// Compute the mean of y.
fn mean(y: &[f64]) -> f64 {
    if y.is_empty() {
        return 0.0;
    }
    let sum: f64 = y.iter().sum();
    sum / (y.len() as f64)
}

/// Compute variance of y.
fn variance(y: &[f64]) -> f64 {
    if y.len() <= 1 {
        return 0.0;
    }
    let m = mean(y);
    let mut var = 0.0;
    for &val in y {
        let diff = val - m;
        var += diff * diff;
    }
    var / (y.len() as f64)
}

/// Compute variance for a subset of indices in y.
fn variance_subset(y: &[f64], indices: &[usize]) -> f64 {
    if indices.len() <= 1 {
        return 0.0;
    }
    let m = indices.iter().map(|&i| y[i]).sum::<f64>() / (indices.len() as f64);
    let mut var = 0.0;
    for &i in indices {
        let diff = y[i] - m;
        var += diff * diff;
    }
    var / (indices.len() as f64)
}

// ---- GradientBoostedModel Implementation ----

impl GradientBoostedModel {
    /// Create a new GradientBoostedModel with the specified objective and config.
    pub fn new(objective: GBMObjective, config: GBMConfig) -> Self {
        Self {
            trees: Vec::new(),
            objective,
            config,
            init_pred: 0.0,
        }
    }

    /// Fit the gradient boosting model to `features` and `labels`.
    /// For MSE, `labels` can be any real values.
    /// For BinaryLogistic, `labels` must be 0.0 or 1.0.
    pub fn fit(&mut self, features: &[Vec<f64>], labels: &[f64]) {
        let n = features.len();
        if n == 0 {
            panic!("No training data provided.");
        }
        if labels.len() != n {
            panic!("Features and labels must have same length.");
        }

        // Validate labels for logistic
        if let GBMObjective::BinaryLogistic = self.objective {
            for &lbl in labels {
                if !(0.0..=1.0).contains(&lbl) {
                    panic!(
                        "For BinaryLogistic, labels must be in {{0.0, 1.0}}. Found {}",
                        lbl
                    );
                }
            }
        }

        // Initialize model
        match self.objective {
            GBMObjective::MSE => {
                // typically init_pred = mean of y
                self.init_pred = mean(labels);
            }
            GBMObjective::BinaryLogistic => {
                // For binary logistic, initialize based on class balance
                let pos_count = labels.iter().filter(|&&x| x > 0.5).count() as f64;
                let n = labels.len() as f64;

                // Handle edge cases explicitly
                self.init_pred = if pos_count == 0.0 {
                    -10.0 // large negative number to ensure initial predictions are close to 0
                } else if pos_count == n {
                    10.0 // large positive number to ensure initial predictions are close to 1
                } else {
                    // Initialize with log(p/(1-p)) where p is the proportion of positive class
                    let p = pos_count / n;
                    (p / (1.0 - p)).ln()
                };
            }
        }

        let mut current_pred = vec![self.init_pred; n]; // F_{m-1}(x_i)

        self.trees.clear();
        for _ in 0..self.config.n_estimators {
            match self.objective {
                GBMObjective::MSE => {
                    // r_i = y_i - F_{m-1}(x_i)
                    let residuals: Vec<f64> = labels
                        .iter()
                        .zip(&current_pred)
                        .map(|(&y_i, &f_i)| y_i - f_i)
                        .collect();

                    // Fit unweighted tree to residuals
                    let mut tree = DecisionTreeRegressor::new(
                        self.config.max_depth,
                        self.config.min_samples_split,
                    );
                    tree.fit(features, &residuals);

                    // Get predictions before moving tree
                    let predictions: Vec<_> =
                        features.iter().map(|x| tree.predict_one(x)).collect();

                    // Store tree and update predictions
                    self.trees.push(tree);
                    for i in 0..n {
                        current_pred[i] += self.config.learning_rate * predictions[i];
                    }
                }
                GBMObjective::BinaryLogistic => {
                    // Current predictions -> probabilities
                    let p: Vec<f64> = current_pred
                        .iter()
                        .map(|&f_i| 1.0 / (1.0 + (-f_i).exp()))
                        .collect();

                    // First-order gradient: r_i = y_i - p_i
                    let residuals: Vec<f64> = labels
                        .iter()
                        .zip(&p)
                        .map(|(&y_i, &p_i)| y_i - p_i)
                        .collect();

                    // Second-order gradient (Hessian): w_i = p_i(1 - p_i)
                    let weights: Vec<f64> = p.iter().map(|&p_i| p_i * (1.0 - p_i)).collect();

                    // Compute z_i = r_i / w_i for the tree target
                    let z: Vec<f64> = residuals
                        .iter()
                        .zip(&weights)
                        .map(
                            |(&r_i, &w_i)| {
                                if w_i.abs() < 1e-15 {
                                    0.0
                                } else {
                                    r_i / w_i
                                }
                            },
                        )
                        .collect();

                    // Fit weighted regression tree and get predictions
                    let mut tree = WeightedDecisionTreeRegressor::new(
                        self.config.max_depth,
                        self.config.min_samples_split,
                    );
                    tree.fit(features, &z, &weights);

                    // Get predictions and convert tree
                    let predictions: Vec<_> =
                        features.iter().map(|x| tree.predict_one(x)).collect();

                    // Convert to unweighted tree and store
                    self.trees.push(tree.into_unweighted());

                    // Update predictions
                    for i in 0..n {
                        current_pred[i] += self.config.learning_rate * predictions[i];
                    }
                }
            }
        }
    }

    /// Predict (regression or classification) for a single feature vector.
    /// - For MSE, returns the raw boosted prediction.
    /// - For BinaryLogistic, returns the class label 0.0 or 1.0, using threshold 0.5 on sigmoid.
    pub fn predict_one(&self, sample: &[f64]) -> f64 {
        let mut score = self.init_pred;
        for tree in &self.trees {
            score += self.config.learning_rate * tree.predict_one(sample);
        }
        match self.objective {
            GBMObjective::MSE => score,
            GBMObjective::BinaryLogistic => {
                // logistic transform => class
                let prob = 1.0 / (1.0 + (-score).exp());
                if prob >= 0.5 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    /// Predict multiple samples at once.
    pub fn predict_batch(&self, data: &[Vec<f64>]) -> Vec<f64> {
        data.iter().map(|row| self.predict_one(row)).collect()
    }

    /// Returns raw decision function values:
    ///   - For MSE, identical to `predict_one`.
    ///   - For BinaryLogistic, returns the log-odds (score) before sigmoid.
    pub fn decision_function_one(&self, sample: &[f64]) -> f64 {
        let mut score = self.init_pred;
        for tree in &self.trees {
            score += self.config.learning_rate * tree.predict_one(sample);
        }
        score
    }
}

/// A decision tree regressor that supports sample weights during training
#[derive(Clone)]
struct WeightedDecisionTreeRegressor {
    root: Option<Box<TreeNode>>,
    max_depth: usize,
    min_samples_split: usize,
}

impl WeightedDecisionTreeRegressor {
    fn new(max_depth: usize, min_samples_split: usize) -> Self {
        Self {
            root: None,
            max_depth,
            min_samples_split,
        }
    }

    fn weighted_mean(z: &[f64], w: &[f64]) -> f64 {
        let wsum: f64 = w.iter().sum();
        if wsum <= 1e-12 {
            return 0.0;
        }
        z.iter()
            .zip(w.iter())
            .map(|(&zi, &wi)| zi * wi)
            .sum::<f64>()
            / wsum
    }

    fn weighted_variance(z: &[f64], w: &[f64]) -> f64 {
        let wsum: f64 = w.iter().sum();
        if wsum <= 1e-12 {
            return 0.0;
        }
        let mean_w = Self::weighted_mean(z, w);

        let mut var = 0.0;
        for (&zi, &wi) in z.iter().zip(w.iter()) {
            let diff = zi - mean_w;
            var += wi * diff * diff;
        }
        var / wsum
    }

    fn find_best_split(
        &self,
        features: &[Vec<f64>],
        z: &[f64],
        w: &[f64],
        feature_indices: &[usize],
        sample_indices: &[usize],
    ) -> Option<(usize, f64, Vec<usize>, Vec<usize>)> {
        let n_samples = sample_indices.len();
        if n_samples < self.min_samples_split {
            return None;
        }

        let current_var = Self::weighted_variance(
            &sample_indices.iter().map(|&i| z[i]).collect::<Vec<_>>(),
            &sample_indices.iter().map(|&i| w[i]).collect::<Vec<_>>(),
        );

        let mut best_gain = 0.0;
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_left = Vec::new();
        let mut best_right = Vec::new();

        for &feature_idx in feature_indices {
            // Sort samples by feature value
            let mut sorted_indices = sample_indices.to_vec();
            sorted_indices.sort_by(|&a, &b| {
                features[a][feature_idx]
                    .partial_cmp(&features[b][feature_idx])
                    .unwrap()
            });

            // Try all possible splits
            for i in 0..(n_samples - 1) {
                let threshold = (features[sorted_indices[i]][feature_idx]
                    + features[sorted_indices[i + 1]][feature_idx])
                    / 2.0;

                let (left, right): (Vec<_>, Vec<_>) = sorted_indices
                    .iter()
                    .copied()
                    .partition(|&idx| features[idx][feature_idx] <= threshold);

                // Skip if either side is empty
                if left.is_empty() || right.is_empty() {
                    continue;
                }

                let left_var = Self::weighted_variance(
                    &left.iter().map(|&i| z[i]).collect::<Vec<_>>(),
                    &left.iter().map(|&i| w[i]).collect::<Vec<_>>(),
                );
                let right_var = Self::weighted_variance(
                    &right.iter().map(|&i| z[i]).collect::<Vec<_>>(),
                    &right.iter().map(|&i| w[i]).collect::<Vec<_>>(),
                );

                let left_weight: f64 = left.iter().map(|&i| w[i]).sum();
                let right_weight: f64 = right.iter().map(|&i| w[i]).sum();
                let total_weight = left_weight + right_weight;

                let gain = current_var
                    - (left_weight * left_var + right_weight * right_var) / total_weight;

                if gain > best_gain {
                    best_gain = gain;
                    best_feature = feature_idx;
                    best_threshold = threshold;
                    best_left = left;
                    best_right = right;
                }
            }
        }

        if best_gain > 0.0 {
            Some((best_feature, best_threshold, best_left, best_right))
        } else {
            None
        }
    }

    fn build_tree(
        &self,
        features: &[Vec<f64>],
        z: &[f64],
        w: &[f64],
        feature_indices: &[usize],
        sample_indices: &[usize],
        depth: usize,
    ) -> Box<TreeNode> {
        // If max depth reached or no split found, create leaf
        if depth >= self.max_depth || sample_indices.len() < self.min_samples_split {
            return Box::new(TreeNode::Leaf(Self::weighted_mean(
                &sample_indices.iter().map(|&i| z[i]).collect::<Vec<_>>(),
                &sample_indices.iter().map(|&i| w[i]).collect::<Vec<_>>(),
            )));
        }

        // Try to find best split
        if let Some((feature, threshold, left_indices, right_indices)) =
            self.find_best_split(features, z, w, feature_indices, sample_indices)
        {
            let left = self.build_tree(features, z, w, feature_indices, &left_indices, depth + 1);
            let right = self.build_tree(features, z, w, feature_indices, &right_indices, depth + 1);

            Box::new(TreeNode::Internal {
                feature_index: feature,
                threshold,
                left,
                right,
            })
        } else {
            // No good split found, create leaf
            Box::new(TreeNode::Leaf(Self::weighted_mean(
                &sample_indices.iter().map(|&i| z[i]).collect::<Vec<_>>(),
                &sample_indices.iter().map(|&i| w[i]).collect::<Vec<_>>(),
            )))
        }
    }

    fn fit(&mut self, features: &[Vec<f64>], z: &[f64], w: &[f64]) {
        let n_samples = features.len();
        let n_features = features[0].len();

        let feature_indices: Vec<_> = (0..n_features).collect();
        let sample_indices: Vec<_> = (0..n_samples).collect();

        self.root = Some(self.build_tree(features, z, w, &feature_indices, &sample_indices, 0));
    }

    fn predict_one(&self, features: &[f64]) -> f64 {
        let node = self.root.as_ref().unwrap();
        Self::predict_one_recursive(node, features)
    }

    fn predict_one_recursive(node: &TreeNode, features: &[f64]) -> f64 {
        match node {
            TreeNode::Leaf(value) => *value,
            TreeNode::Internal {
                feature_index,
                threshold,
                left,
                right,
            } => {
                if features[*feature_index] <= *threshold {
                    Self::predict_one_recursive(left, features)
                } else {
                    Self::predict_one_recursive(right, features)
                }
            }
        }
    }

    fn into_unweighted(self) -> DecisionTreeRegressor {
        DecisionTreeRegressor {
            root: match self.root {
                Some(root) => *root,
                None => TreeNode::Leaf(0.0),
            },
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gbm_regression() {
        // We'll fit a simple y = x1 + x2 problem, with some noise
        let x = vec![
            vec![0.0, 0.0],
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![3.0, 4.0],
            vec![4.0, 3.0],
        ];
        let y = x.iter().map(|row| row[0] + row[1]).collect::<Vec<_>>();

        let config = GBMConfig {
            n_estimators: 50,
            learning_rate: 0.1,
            max_depth: 2,
            min_samples_split: 2,
            seed: Some(42),
        };
        let mut model = GradientBoostedModel::new(GBMObjective::MSE, config);
        model.fit(&x, &y);

        // Check predictions on training data
        for (i, row) in x.iter().enumerate() {
            let pred = model.predict_one(row);
            let true_val = y[i];
            let err = (pred - true_val).abs();
            // We expect small error
            assert!(err < 0.5, "Expected small error, got err={}", err);
        }
    }

    #[test]
    fn test_gbm_binary_logistic() {
        // Simple classification: label=1 if x1 + x2 > 2, else 0
        let x = vec![
            vec![0.0, 0.0], // 0 (clearly negative)
            vec![0.5, 1.0], // 0 (below boundary)
            vec![1.0, 0.8], // 0 (below boundary)
            vec![1.5, 0.7], // 1 (just above boundary)
            vec![2.0, 1.0], // 1 (above boundary)
            vec![3.0, 2.0], // 1 (clearly positive)
        ];
        let y = x
            .iter()
            .map(|row| if row[0] + row[1] > 2.0 { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();

        let config = GBMConfig {
            n_estimators: 100,   // more trees for better convergence
            learning_rate: 0.05, // smaller learning rate for stability
            max_depth: 3,        // slightly deeper trees
            min_samples_split: 2,
            seed: Some(42),
        };
        let mut model = GradientBoostedModel::new(GBMObjective::BinaryLogistic, config);
        model.fit(&x, &y);

        // Test points that are far from the decision boundary
        let test_points = vec![
            (vec![0.0, 0.0], 0.0), // clearly negative
            (vec![4.0, 4.0], 1.0), // clearly positive
        ];

        for (point, expected) in test_points {
            let pred = model.predict_one(&point);
            assert_eq!(
                pred, expected,
                "Failed to classify clear-cut point {:?}",
                point
            );
        }
    }
}
