use rand::prelude::*;
use std::collections::HashMap;

/// A simple Random Forest classifier based on a CART-like decision tree.
/// For each tree, we perform bootstrap sampling of the training data and
/// random feature subspace selection. Then a majority vote across trees
/// gives the final classification.
///
/// This example only handles classification with discrete labels (hashable types).
///
/// # Type Parameters
/// - `L`: The label type (e.g., `String`, integer, etc.), must implement `Eq + Hash + Clone`.
#[derive(Debug)]
pub struct RandomForest<L> {
    /// The ensemble of decision trees.
    trees: Vec<DecisionTree<L>>,
    /// Number of features considered at each split (if None, uses sqrt(num_features)).
    max_features: Option<usize>,
    /// Number of trees in the forest.
    n_trees: usize,
    /// Ratio of samples used for each tree (bootstrap sampling).
    sample_ratio: f64,
    /// Maximum tree depth (None = unlimited).
    max_depth: Option<usize>,
}

/// A basic decision tree node for classification.
#[derive(Debug, Clone)]
enum TreeNode<L> {
    /// Leaf node with a predicted label.
    Leaf(L),
    /// Internal node splitting on a feature index (for continuous split).
    /// - For discrete (categorical) splits, store each category in `branches`:
    ///   The string is the category value, the node is the subtree.
    /// - For continuous splits, we store two keys: "<=" and ">" in `branches`.
    Internal {
        feature_index: usize,
        threshold: f64,      // only used if continuous
        is_continuous: bool, // true if the feature is treated as continuous
        branches: HashMap<String, TreeNode<L>>,
    },
}

/// A single decision tree in the forest.
#[derive(Debug)]
pub struct DecisionTree<L> {
    root: TreeNode<L>,
}

impl<L: Clone + Eq + std::hash::Hash> RandomForest<L> {
    /// Creates a new RandomForest.
    ///
    /// # Arguments
    /// - `n_trees`: number of trees in the ensemble.
    /// - `max_features`: how many features to consider at each split. If None, defaults to `sqrt(num_features)` for each tree.
    /// - `sample_ratio`: fraction of training set used to build each tree (bootstrap). Usually 1.0 for classic bagging.
    /// - `max_depth`: maximum depth of each tree. If None, unlimited depth is allowed.
    pub fn new(
        n_trees: usize,
        max_features: Option<usize>,
        sample_ratio: f64,
        max_depth: Option<usize>,
    ) -> Self {
        RandomForest {
            trees: Vec::new(),
            max_features,
            n_trees,
            sample_ratio,
            max_depth,
        }
    }

    /// Trains the random forest on the given dataset.
    ///
    /// # Arguments
    /// - `data`: a slice of feature vectors, each feature is a string if categorical or a numeric string if continuous.
    /// - `labels`: corresponding labels for each data row.
    /// - `seed`: RNG seed for reproducibility (optional).
    ///
    /// # Panics
    /// - If `data.len() != labels.len()`.
    /// - If `data` is empty or has inconsistent row lengths.
    pub fn fit(&mut self, data: &[Vec<String>], labels: &[L], seed: Option<u64>) {
        assert_eq!(
            data.len(),
            labels.len(),
            "data and labels must match in length"
        );
        let n_samples = data.len();
        if n_samples == 0 {
            panic!("No training data provided.");
        }
        let num_features = data[0].len();
        for row in data.iter() {
            assert_eq!(
                row.len(),
                num_features,
                "Inconsistent feature dimension in data."
            );
        }

        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let m_features = self.max_features.unwrap_or_else(|| {
            // default is sqrt(num_features), floored
            (num_features as f64).sqrt().floor() as usize
        });

        self.trees.clear();
        self.trees.reserve(self.n_trees);

        for _ in 0..self.n_trees {
            // Bootstrap sample
            let subset_size = (self.sample_ratio * n_samples as f64).round() as usize;
            let (sampled_data, sampled_labels) =
                bootstrap_sample(data, labels, subset_size, &mut rng);

            // Build a decision tree with random feature subspace
            let tree = DecisionTree::build(
                &sampled_data,
                &sampled_labels,
                m_features,
                self.max_depth,
                &mut rng,
            );
            self.trees.push(tree);
        }
    }

    /// Predicts the label for a single feature vector by majority vote of the ensemble.
    ///
    /// # Panics
    /// - If the forest is empty (not trained).
    pub fn predict(&self, features: &[String]) -> L {
        if self.trees.is_empty() {
            panic!("RandomForest has no trained trees.");
        }
        let mut votes = HashMap::new();
        for tree in &self.trees {
            let lbl = tree.predict(features);
            *votes.entry(lbl).or_insert(0) += 1;
        }
        votes.into_iter().max_by_key(|(_k, v)| *v).unwrap().0
    }

    /// Predict multiple rows at once.
    pub fn predict_batch(&self, data: &[Vec<String>]) -> Vec<L> {
        data.iter().map(|row| self.predict(row)).collect()
    }
}

impl<L: Clone + Eq + std::hash::Hash> DecisionTree<L> {
    /// Train a single decision tree using a CART-like approach with random feature subsets.
    ///
    /// # Arguments
    /// - `data`: subset of the training data (rows).
    /// - `labels`: subset of the labels for those rows.
    /// - `max_features`: how many features to consider at each split (randomly chosen from total).
    /// - `max_depth`: maximum recursion depth.
    /// - `rng`: random generator to shuffle/select features.
    fn build(
        data: &[Vec<String>],
        labels: &[L],
        max_features: usize,
        max_depth: Option<usize>,
        rng: &mut impl Rng,
    ) -> Self {
        let root_node = build_tree_recursive(data, labels, max_features, max_depth, 0, rng);
        DecisionTree { root: root_node }
    }

    /// Predict a label for the given feature vector.
    pub fn predict(&self, features: &[String]) -> L {
        traverse(&self.root, features)
    }
}

/// Recursively build a decision tree node.
fn build_tree_recursive<L: Clone + Eq + std::hash::Hash>(
    data: &[Vec<String>],
    labels: &[L],
    max_features: usize,
    max_depth: Option<usize>,
    current_depth: usize,
    rng: &mut impl Rng,
) -> TreeNode<L> {
    // If all labels are the same, return a Leaf.
    if all_same(labels) {
        return TreeNode::Leaf(labels[0].clone());
    }

    // If we've reached max depth or data is too small, return a Leaf with majority label.
    if let Some(md) = max_depth {
        if current_depth >= md {
            return TreeNode::Leaf(majority_label(labels));
        }
    }

    // Attempt to find best split among a random subset of features
    let num_features = data[0].len();
    let feature_indices = random_feature_subset(num_features, max_features, rng);

    let (best_feat, best_threshold, best_gini, left_idx, right_idx) =
        find_best_split(data, labels, &feature_indices);

    // If no improvement in Gini or no valid split, make a Leaf
    if best_gini < 1e-12 || left_idx.is_empty() || right_idx.is_empty() {
        return TreeNode::Leaf(majority_label(labels));
    }

    // Partition data & labels
    let left_data: Vec<Vec<String>> = left_idx.iter().map(|&i| data[i].clone()).collect();
    let left_labels: Vec<L> = left_idx.iter().map(|&i| labels[i].clone()).collect();
    let right_data: Vec<Vec<String>> = right_idx.iter().map(|&i| data[i].clone()).collect();
    let right_labels: Vec<L> = right_idx.iter().map(|&i| labels[i].clone()).collect();

    // Recursively build children
    let left_child = build_tree_recursive(
        &left_data,
        &left_labels,
        max_features,
        max_depth,
        current_depth + 1,
        rng,
    );
    let right_child = build_tree_recursive(
        &right_data,
        &right_labels,
        max_features,
        max_depth,
        current_depth + 1,
        rng,
    );

    // Store branches in a HashMap
    let mut branches = HashMap::new();
    branches.insert("≤".to_string(), left_child);
    branches.insert(">".to_string(), right_child);

    TreeNode::Internal {
        feature_index: best_feat,
        threshold: best_threshold,
        is_continuous: true,
        branches,
    }
}

/// Attempt to find the best split among the given `feature_indices`.
/// This example always treats features as continuous if parseable, otherwise
/// lumps them into a pseudo "ordinal" (still uses a threshold approach).
fn find_best_split<L: Clone + Eq + std::hash::Hash>(
    data: &[Vec<String>],
    labels: &[L],
    feature_indices: &[usize],
) -> (usize, f64, f64, Vec<usize>, Vec<usize>) {
    let mut best_feat = 0;
    let mut best_threshold = 0.0;
    let mut best_gini = 0.0;
    let mut best_left = Vec::new();
    let mut best_right = Vec::new();

    // current overall Gini
    let base_gini = gini(labels);

    // We track the best "reduction" in Gini
    let mut best_reduction = 0.0;

    for &feat in feature_indices {
        // Collect all numeric values. If parse fails, skip
        let mut values = Vec::new();
        for row in data {
            if let Ok(v) = row[feat].parse::<f64>() {
                values.push(v);
            }
        }
        if values.is_empty() {
            continue;
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        values.dedup();

        // We'll generate thresholds from midpoints
        for w in values.windows(2) {
            let th = 0.5 * (w[0] + w[1]);
            let (l_idx, r_idx) = partition_indices(data, feat, th);

            if l_idx.is_empty() || r_idx.is_empty() {
                continue;
            }

            let l_gini = gini_of_subset(labels, &l_idx);
            let r_gini = gini_of_subset(labels, &r_idx);
            let nl = l_idx.len() as f64;
            let nr = r_idx.len() as f64;
            let n = data.len() as f64;
            // Weighted Gini
            let split_gini = (nl / n) * l_gini + (nr / n) * r_gini;

            let reduction = base_gini - split_gini;
            if reduction > best_reduction {
                best_reduction = reduction;
                best_feat = feat;
                best_threshold = th;
                best_gini = split_gini; // final gini of the node
                best_left = l_idx;
                best_right = r_idx;
            }
        }
    }

    (best_feat, best_threshold, best_gini, best_left, best_right)
}

/// Partition data indices into left or right based on `data[i][feat] <= threshold` for numeric,
/// or based on string comparison for categorical.
fn partition_indices(
    data: &[Vec<String>],
    feat: usize,
    threshold: f64,
) -> (Vec<usize>, Vec<usize>) {
    let mut left_idx = Vec::new();
    let mut right_idx = Vec::new();

    // First check if all values in this feature can be parsed as numbers
    let mut all_numeric = true;
    let mut any_numeric = false;
    for row in data.iter() {
        if row[feat].parse::<f64>().is_ok() {
            any_numeric = true;
        } else {
            all_numeric = false;
            break;
        }
    }

    // If all values are numeric, use numeric comparison
    if all_numeric {
        for (i, row) in data.iter().enumerate() {
            let val = row[feat].parse::<f64>().unwrap();
            if val <= threshold {
                left_idx.push(i);
            } else {
                right_idx.push(i);
            }
        }
    }
    // If some values are numeric but not all, try to parse each and compare
    else if any_numeric {
        for (i, row) in data.iter().enumerate() {
            // Always try numeric comparison first for consistency
            if let Ok(val) = row[feat].parse::<f64>() {
                if val <= threshold {
                    left_idx.push(i);
                } else {
                    right_idx.push(i);
                }
            } else {
                // Only fall back to string comparison if parse fails
                let val_str = row[feat].trim();
                if val_str <= threshold.to_string().as_str() {
                    left_idx.push(i);
                } else {
                    right_idx.push(i);
                }
            }
        }
    }
    // If no numeric values, use string comparison
    else {
        let threshold_str = threshold.to_string();
        for (i, row) in data.iter().enumerate() {
            let val_str = row[feat].trim();
            if val_str <= threshold_str.as_str() {
                left_idx.push(i);
            } else {
                right_idx.push(i);
            }
        }
    }

    (left_idx, right_idx)
}

/// Compute the Gini impurity of an entire label list.
fn gini<L: Clone + Eq + std::hash::Hash>(labels: &[L]) -> f64 {
    let mut counts = HashMap::new();
    for lbl in labels {
        *counts.entry(lbl.clone()).or_insert(0) += 1;
    }
    let n = labels.len() as f64;
    let mut impurity = 1.0;
    for (_lbl, c) in counts {
        let p = c as f64 / n;
        impurity -= p * p;
    }
    impurity
}

/// Compute Gini impurity for a subset of `labels` given by `indices`.
fn gini_of_subset<L: Clone + Eq + std::hash::Hash>(labels: &[L], indices: &[usize]) -> f64 {
    let mut counts = HashMap::new();
    for &i in indices {
        *counts.entry(labels[i].clone()).or_insert(0) += 1;
    }
    let n = indices.len() as f64;
    let mut impurity = 1.0;
    for (_lbl, c) in counts {
        let p = c as f64 / n;
        impurity -= p * p;
    }
    impurity
}

/// Check if all elements in `labels` are the same.
fn all_same<L: PartialEq>(labels: &[L]) -> bool {
    if labels.is_empty() {
        return true;
    }
    let first = &labels[0];
    labels.iter().all(|x| x == first)
}

/// Returns the majority label.
fn majority_label<L: Clone + Eq + std::hash::Hash>(labels: &[L]) -> L {
    let mut counts = HashMap::new();
    for lbl in labels {
        *counts.entry(lbl.clone()).or_insert(0) += 1;
    }
    counts.into_iter().max_by_key(|(_, c)| *c).unwrap().0
}

/// Traverse the tree to classify a feature vector.
fn traverse<L: Clone>(node: &TreeNode<L>, features: &[String]) -> L {
    match node {
        TreeNode::Leaf(lbl) => lbl.clone(),
        TreeNode::Internal {
            feature_index,
            threshold,
            is_continuous,
            branches,
        } => {
            if *is_continuous {
                let val = features[*feature_index]
                    .parse::<f64>()
                    .unwrap_or(f64::INFINITY);
                if val <= *threshold {
                    match branches.get("≤") {
                        Some(child) => traverse(child, features),
                        None => panic!("Missing ≤ branch in a continuous split"),
                    }
                } else {
                    match branches.get(">") {
                        Some(child) => traverse(child, features),
                        None => panic!("Missing > branch in a continuous split"),
                    }
                }
            } else {
                // For purely categorical scenario (not used in this example),
                // we'd do something like:
                let feat_val = &features[*feature_index];
                match branches.get(feat_val) {
                    Some(child) => traverse(child, features),
                    None => {
                        // fallback or majority child if unknown category
                        let mut iter = branches.values();
                        traverse(iter.next().unwrap(), features)
                    }
                }
            }
        }
    }
}

/// Returns a random subset of feature indices of size up to `max_features`.
fn random_feature_subset(
    num_features: usize,
    max_features: usize,
    rng: &mut impl Rng,
) -> Vec<usize> {
    // sample without replacement
    let mut all_feats: Vec<usize> = (0..num_features).collect();
    all_feats.shuffle(rng);
    all_feats.truncate(max_features.min(num_features));
    all_feats
}

/// Performs bootstrap sampling of size `sample_size` from the dataset, possibly with replacement.
fn bootstrap_sample<L: Clone>(
    data: &[Vec<String>],
    labels: &[L],
    sample_size: usize,
    rng: &mut impl Rng,
) -> (Vec<Vec<String>>, Vec<L>) {
    let n = data.len();
    let mut out_data = Vec::with_capacity(sample_size);
    let mut out_labels = Vec::with_capacity(sample_size);
    for _ in 0..sample_size {
        let idx = rng.gen_range(0..n);
        out_data.push(data[idx].clone());
        out_labels.push(labels[idx].clone());
    }
    (out_data, out_labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_forest_basic() {
        // Simple synthetic dataset
        // We'll define x ~ 2D with numeric strings: x1, x2
        // label = "A" if x1 + x2 < 5, else "B"
        let data = vec![
            vec!["1".to_string(), "1".to_string()], // sum=2, A
            vec!["2".to_string(), "2".to_string()], // sum=4, A
            vec!["3".to_string(), "3".to_string()], // sum=6, B
            vec!["4".to_string(), "4".to_string()], // sum=8, B
            vec!["1".to_string(), "3".to_string()], // sum=4, A
            vec!["3".to_string(), "2".to_string()], // sum=5, B
            vec!["2".to_string(), "1".to_string()], // sum=3, A
            vec!["4".to_string(), "3".to_string()], // sum=7, B
            vec!["0".to_string(), "4".to_string()], // sum=4, A
            vec!["4".to_string(), "2".to_string()], // sum=6, B
            // Add more points to reinforce the boundary
            vec!["0".to_string(), "2".to_string()], // sum=2, A
            vec!["5".to_string(), "3".to_string()], // sum=8, B
            vec!["2".to_string(), "2.5".to_string()], // sum=4.5, A
            vec!["3".to_string(), "4".to_string()], // sum=7, B
            vec!["1".to_string(), "2".to_string()], // sum=3, A
            vec!["6".to_string(), "2".to_string()], // sum=8, B
        ];
        let labels = vec![
            "A", "A", "B", "B", "A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "A", "B",
        ];

        // Use more trees (100) and limit depth to 5 to prevent overfitting
        let mut rf = RandomForest::new(100, None, 1.0, Some(5));
        rf.fit(&data, &labels, Some(42));

        // Test with extreme points
        let test1 = vec!["0".to_string(), "0".to_string()]; // sum=0 => definitely "A"
        let test2 = vec!["9".to_string(), "9".to_string()]; // sum=18 => definitely "B"

        let pred1 = rf.predict(&test1);
        let pred2 = rf.predict(&test2);

        assert_eq!(
            pred1, "A",
            "Test point [0,0] was classified as {} but should be A",
            pred1
        );
        assert_eq!(
            pred2, "B",
            "Test point [9,9] was classified as {} but should be B",
            pred2
        );
    }
}
