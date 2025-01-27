use std::collections::HashMap;

/// Configuration for building a decision tree.
#[derive(Debug, Clone)]
pub enum DecisionTreeAlgorithm {
    /// ID3 algorithm (uses information gain). Typically expects discrete features only.
    ID3,
    /// C4.5 algorithm (uses gain ratio). Supports both discrete and continuous features.
    C45,
}

/// Represents a trained decision tree node.
#[derive(Debug, Clone)]
pub enum DecisionTreeNode<L> {
    /// An internal node that splits on a feature (categorical or continuous).
    Internal {
        /// The index of the feature used for splitting.
        feature_index: usize,
        /// If `split_value` is `Some(x)`, this is a continuous split:
        ///   - left branch: feature <= x
        ///   - right branch: feature > x
        ///     If `None`, this is a categorical split with multiple branches in `branches`.
        split_value: Option<f64>,
        /// For categorical splits, each branch is keyed by a feature value.
        /// For continuous splits, we only use `branches[0]` as the "left" and `branches[1]` as the "right".
        branches: HashMap<String, DecisionTreeNode<L>>,
    },
    /// A leaf node with a predicted label.
    Leaf(L),
}

/// A decision tree classifier that can be trained using ID3 or C4.5.
#[derive(Debug, Clone)]
pub struct DecisionTree<L> {
    pub root: DecisionTreeNode<L>,
    pub algorithm: DecisionTreeAlgorithm,
}

impl<L: Clone + Eq + std::hash::Hash> DecisionTree<L> {
    /// Builds a decision tree from the provided dataset.
    ///
    /// - `data`: a slice of feature vectors, each element is a vector of strings for categorical
    ///   features or "numeric" strings for continuous. (C4.5 attempts to interpret numeric fields.)
    /// - `labels`: the corresponding label for each data row.
    /// - `feature_names`: names or identifiers for each column in `data` (optional, but helps debug).
    /// - `algorithm`: ID3 or C4.5.
    ///
    /// # Panics
    /// - If `data.len() != labels.len()`.
    /// - If `data` is empty or has inconsistent row lengths.
    pub fn fit(
        data: &[Vec<String>],
        labels: &[L],
        feature_names: Option<&[String]>,
        algorithm: DecisionTreeAlgorithm,
    ) -> Self {
        assert_eq!(
            data.len(),
            labels.len(),
            "data and labels must match in length"
        );
        if data.is_empty() {
            panic!("No training data provided.");
        }
        let num_features = data[0].len();
        for (i, row) in data.iter().enumerate() {
            if row.len() != num_features {
                panic!("Row {} has inconsistent feature length.", i);
            }
        }

        let feat_names = match feature_names {
            Some(names) => {
                assert_eq!(
                    names.len(),
                    num_features,
                    "feature_names must match data columns"
                );
                names.to_vec()
            }
            None => (0..num_features).map(|i| format!("F{}", i)).collect(),
        };

        let root = build_tree(data, labels, &feat_names, &algorithm);
        Self { root, algorithm }
    }

    /// Predicts the label for a single feature vector (each feature is a string;
    /// numeric features are expected to be parseable as `f64` if the node is a C4.5 continuous split).
    pub fn predict(&self, features: &[String]) -> L {
        traverse_tree(&self.root, features)
    }

    /// Predict for multiple rows.
    pub fn predict_batch(&self, data: &[Vec<String>]) -> Vec<L> {
        data.iter().map(|row| self.predict(row)).collect()
    }
}

/// Recursively builds a decision tree using ID3 or C4.5.
fn build_tree<L: Clone + Eq + std::hash::Hash>(
    data: &[Vec<String>],
    labels: &[L],
    feature_names: &[String],
    algo: &DecisionTreeAlgorithm,
) -> DecisionTreeNode<L> {
    // Check if all labels are the same => Leaf node
    if all_same(labels) {
        return DecisionTreeNode::Leaf(labels[0].clone());
    }

    // If no features left or data is empty => Leaf with most common label
    if data.is_empty() || data[0].is_empty() || feature_names.is_empty() {
        return DecisionTreeNode::Leaf(majority_label(labels));
    }

    // Find the best feature (and split) based on the algorithm
    match algo {
        DecisionTreeAlgorithm::ID3 => {
            let (best_feat_idx, best_split) = find_best_split_id3(data, labels);
            if let Some(split_val) = best_split {
                // Continuous split
                make_continuous_node(data, labels, feature_names, best_feat_idx, split_val, algo)
            } else {
                // Categorical split
                make_categorical_node(data, labels, feature_names, best_feat_idx, algo)
            }
        }
        DecisionTreeAlgorithm::C45 => {
            let (best_feat_idx, best_split) = find_best_split_c45(data, labels);
            if let Some(split_val) = best_split {
                // Continuous split
                make_continuous_node(data, labels, feature_names, best_feat_idx, split_val, algo)
            } else {
                // Categorical split
                make_categorical_node(data, labels, feature_names, best_feat_idx, algo)
            }
        }
    }
}

/// Create a node that splits continuously on `split_value`.
fn make_continuous_node<L: Clone + Eq + std::hash::Hash>(
    data: &[Vec<String>],
    labels: &[L],
    feature_names: &[String],
    feat_idx: usize,
    split_value: f64,
    algo: &DecisionTreeAlgorithm,
) -> DecisionTreeNode<L> {
    // Partition data into <= split_value and > split_value
    let mut left_data = Vec::new();
    let mut left_labels = Vec::new();
    let mut right_data = Vec::new();
    let mut right_labels = Vec::new();

    for (row, lbl) in data.iter().zip(labels.iter()) {
        let val = row[feat_idx].parse::<f64>().unwrap_or(f64::NAN);
        if val.is_nan() {
            continue; // skip or treat as special category
        }
        if val <= split_value {
            left_data.push(row.clone());
            left_labels.push(lbl.clone());
        } else {
            right_data.push(row.clone());
            right_labels.push(lbl.clone());
        }
    }

    // Remove feature column from subsets if we want to avoid re-splitting on the same attribute
    // C4.5 typically can re-split on the same continuous feature with different thresholds, so we can keep it.
    // But for simplicity, let's keep them. If we wanted to remove, we'd do it here.

    let mut branches = HashMap::new();
    branches.insert(
        "≤".to_string(),
        build_tree(&left_data, &left_labels, feature_names, algo),
    );
    branches.insert(
        ">".to_string(),
        build_tree(&right_data, &right_labels, feature_names, algo),
    );

    DecisionTreeNode::Internal {
        feature_index: feat_idx,
        split_value: Some(split_value),
        branches,
    }
}

/// Create a node that splits categorically on each distinct value of the chosen feature.
fn make_categorical_node<L: Clone + Eq + std::hash::Hash>(
    data: &[Vec<String>],
    labels: &[L],
    feature_names: &[String],
    feat_idx: usize,
    algo: &DecisionTreeAlgorithm,
) -> DecisionTreeNode<L> {
    // Group rows by feature value
    let mut subsets: HashMap<String, (Vec<Vec<String>>, Vec<L>)> = HashMap::new();
    for (row, lbl) in data.iter().zip(labels.iter()) {
        let val = row[feat_idx].clone();
        subsets.entry(val).or_default().0.push(row.clone());
        subsets
            .entry(row[feat_idx].clone())
            .or_default()
            .1
            .push(lbl.clone());
    }

    // Possibly remove the feature column if we don't want to reuse the same attribute
    // For ID3 typically we remove the used attribute from feature space.
    // For C4.5 we also remove it if it's categorical. Continuous can be used multiple times.
    // Let's remove it for categorical splits.
    let is_categorical = true; // by definition here
    let updated_feature_names = if is_categorical {
        // remove feat_idx
        let mut new_names = feature_names.to_vec();
        new_names.remove(feat_idx);
        new_names
    } else {
        feature_names.to_vec()
    };

    let mut branches = HashMap::new();
    for (val, (sub_data, sub_labels)) in subsets.into_iter() {
        let next_data = if is_categorical {
            remove_column(&sub_data, feat_idx)
        } else {
            sub_data
        };

        let child = build_tree(&next_data, &sub_labels, &updated_feature_names, algo);
        branches.insert(val, child);
    }

    DecisionTreeNode::Internal {
        feature_index: feat_idx,
        split_value: None,
        branches,
    }
}

/// ID3: choose feature that maximizes information gain. For continuous features, test possible splits.
fn find_best_split_id3<L: Clone + Eq + std::hash::Hash>(
    data: &[Vec<String>],
    labels: &[L],
) -> (usize, Option<f64>) {
    // Evaluate each feature; for continuous features, check candidate splits, pick best.
    // We'll do a simple approach: if a column is parseable as f64 for all rows, treat it as continuous.
    let num_features = data[0].len();
    let base_entropy = entropy(labels);

    let mut best_gain = f64::NEG_INFINITY;
    let mut best_feat = 0;
    let mut best_split: Option<f64> = None;

    for feat_idx in 0..num_features {
        if is_continuous_column(data, feat_idx) {
            // Evaluate all possible thresholds
            let thresholds = possible_thresholds(data, feat_idx);
            for &th in &thresholds {
                let gain = info_gain_continuous(data, labels, feat_idx, th, base_entropy);
                if gain > best_gain {
                    best_gain = gain;
                    best_feat = feat_idx;
                    best_split = Some(th);
                }
            }
        } else {
            let gain = info_gain_categorical(data, labels, feat_idx, base_entropy);
            if gain > best_gain {
                best_gain = gain;
                best_feat = feat_idx;
                best_split = None;
            }
        }
    }
    (best_feat, best_split)
}

/// C4.5: choose feature that maximizes gain ratio. For continuous features, test possible splits similarly.
fn find_best_split_c45<L: Clone + Eq + std::hash::Hash>(
    data: &[Vec<String>],
    labels: &[L],
) -> (usize, Option<f64>) {
    let num_features = data[0].len();
    let base_entropy = entropy(labels);

    let mut best_ratio = f64::NEG_INFINITY;
    let mut best_feat = 0;
    let mut best_split: Option<f64> = None;

    for feat_idx in 0..num_features {
        if is_continuous_column(data, feat_idx) {
            let thresholds = possible_thresholds(data, feat_idx);
            for &th in &thresholds {
                let gain = info_gain_continuous(data, labels, feat_idx, th, base_entropy);
                if gain <= 0.0 {
                    continue;
                }
                let split_info = split_info_continuous(data, feat_idx, th);
                let ratio = if split_info.abs() < 1e-12 {
                    0.0
                } else {
                    gain / split_info
                };
                if ratio > best_ratio {
                    best_ratio = ratio;
                    best_feat = feat_idx;
                    best_split = Some(th);
                }
            }
        } else {
            let gain = info_gain_categorical(data, labels, feat_idx, base_entropy);
            if gain <= 0.0 {
                continue;
            }
            let split_info = split_info_categorical(data, feat_idx);
            let ratio = if split_info.abs() < 1e-12 {
                0.0
            } else {
                gain / split_info
            };
            if ratio > best_ratio {
                best_ratio = ratio;
                best_feat = feat_idx;
                best_split = None;
            }
        }
    }
    (best_feat, best_split)
}

/// Compute info gain for a continuous split on `threshold`.
fn info_gain_continuous<L: Clone + Eq + std::hash::Hash>(
    data: &[Vec<String>],
    labels: &[L],
    feat_idx: usize,
    threshold: f64,
    base_entropy: f64,
) -> f64 {
    let mut left_labels = Vec::new();
    let mut right_labels = Vec::new();
    for (row, lbl) in data.iter().zip(labels.iter()) {
        let val = row[feat_idx].parse::<f64>().unwrap_or(f64::NAN);
        if val.is_nan() {
            continue; // skip or treat as separate
        }
        if val <= threshold {
            left_labels.push(lbl.clone());
        } else {
            right_labels.push(lbl.clone());
        }
    }
    let n = (left_labels.len() + right_labels.len()) as f64;
    let h_left = entropy(&left_labels);
    let h_right = entropy(&right_labels);
    let w_left = left_labels.len() as f64 / n;
    let w_right = right_labels.len() as f64 / n;

    base_entropy - (w_left * h_left + w_right * h_right)
}

/// Compute info gain for a categorical feature.
fn info_gain_categorical<L: Clone + Eq + std::hash::Hash>(
    data: &[Vec<String>],
    labels: &[L],
    feat_idx: usize,
    base_entropy: f64,
) -> f64 {
    let mut subsets: HashMap<String, Vec<L>> = HashMap::new();
    for (row, lbl) in data.iter().zip(labels.iter()) {
        subsets
            .entry(row[feat_idx].clone())
            .or_default()
            .push(lbl.clone());
    }
    let n = labels.len() as f64;
    let mut remainder = 0.0;
    for (_val, sub_labels) in subsets.into_iter() {
        let w = sub_labels.len() as f64 / n;
        remainder += w * entropy(&sub_labels);
    }
    base_entropy - remainder
}

/// Compute split info for a continuous split (C4.5).
/// SplitInfo = - ( (m/n) * log2(m/n) + (n-m)/n * log2((n-m)/n) ) ignoring empty side.
fn split_info_continuous(data: &[Vec<String>], feat_idx: usize, threshold: f64) -> f64 {
    let mut left_count = 0;
    let mut right_count = 0;
    for row in data {
        let val = row[feat_idx].parse::<f64>().unwrap_or(f64::NAN);
        if !val.is_nan() {
            if val <= threshold {
                left_count += 1;
            } else {
                right_count += 1;
            }
        }
    }
    let n = (left_count + right_count) as f64;
    let mut si = 0.0;
    if left_count > 0 {
        let p = left_count as f64 / n;
        si -= p * log2(p);
    }
    if right_count > 0 {
        let p = right_count as f64 / n;
        si -= p * log2(p);
    }
    si
}

/// Compute split info for a categorical feature (C4.5).
fn split_info_categorical(data: &[Vec<String>], feat_idx: usize) -> f64 {
    let mut counts = HashMap::new();
    for row in data {
        *counts.entry(row[feat_idx].clone()).or_insert(0) += 1;
    }
    let n = data.len() as f64;
    let mut si = 0.0;
    for (_val, count) in counts {
        let p = count as f64 / n;
        si -= p * log2(p);
    }
    si
}

/// Find possible thresholds for a continuous column by taking midpoints of sorted unique values.
fn possible_thresholds(data: &[Vec<String>], feat_idx: usize) -> Vec<f64> {
    let mut vals = Vec::new();
    for row in data {
        if let Ok(x) = row[feat_idx].parse::<f64>() {
            if !x.is_nan() {
                vals.push(x);
            }
        }
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    vals.dedup();
    let mut thresholds = Vec::new();
    for w in vals.windows(2) {
        let mid = 0.5 * (w[0] + w[1]);
        thresholds.push(mid);
    }
    thresholds
}

/// Check if a column can be parsed as continuous (f64) for all non-missing entries.
fn is_continuous_column(data: &[Vec<String>], feat_idx: usize) -> bool {
    let mut all_numeric = true;
    for row in data {
        if row[feat_idx].parse::<f64>().is_err() {
            all_numeric = false;
            break;
        }
    }
    all_numeric
}

/// Remove the column at `col_idx` from each row.
fn remove_column(data: &[Vec<String>], col_idx: usize) -> Vec<Vec<String>> {
    let mut out = Vec::new();
    for row in data {
        let mut new_row = row.clone();
        new_row.remove(col_idx);
        out.push(new_row);
    }
    out
}

/// Traverse the decision tree to classify a feature vector.
fn traverse_tree<L: Clone>(node: &DecisionTreeNode<L>, features: &[String]) -> L {
    match node {
        DecisionTreeNode::Leaf(lbl) => lbl.clone(),
        DecisionTreeNode::Internal {
            feature_index,
            split_value,
            branches,
        } => {
            if let Some(th) = split_value {
                // Continuous
                let val = features[*feature_index].parse::<f64>().unwrap_or(f64::NAN);
                let branch_key = if val <= *th { "≤" } else { ">" };
                match branches.get(branch_key) {
                    Some(next_node) => traverse_tree(next_node, features),
                    None => {
                        // fallback if something is missing
                        // e.g., val is NaN => choose majority child or any child
                        // just pick the first or a default.
                        let mut iter = branches.values();
                        iter.next().unwrap().clone_leaf()
                    }
                }
            } else {
                // Categorical
                let feat_val = &features[*feature_index];
                match branches.get(feat_val) {
                    Some(next_node) => traverse_tree(next_node, features),
                    None => {
                        // fallback if unseen category
                        let mut iter = branches.values();
                        iter.next().unwrap().clone_leaf()
                    }
                }
            }
        }
    }
}

impl<L: Clone> DecisionTreeNode<L> {
    /// Helper to return a leaf label if self is Leaf, otherwise picks
    /// a child's leaf. This is used as a fallback for missing branches.
    fn clone_leaf(&self) -> L {
        match self {
            DecisionTreeNode::Leaf(lbl) => lbl.clone(),
            DecisionTreeNode::Internal { branches, .. } => {
                // Recursively find a leaf in one of the branches
                let first = branches.values().next().expect("No branches in node");
                first.clone_leaf()
            }
        }
    }
}

/// Returns true if all elements in `labels` are the same.
fn all_same<L: PartialEq>(labels: &[L]) -> bool {
    if labels.is_empty() {
        return true;
    }
    labels.iter().all(|x| x == &labels[0])
}

/// Return the label that appears most frequently in `labels`.
fn majority_label<L: Clone + Eq + std::hash::Hash>(labels: &[L]) -> L {
    let mut counts = HashMap::new();
    for lbl in labels {
        *counts.entry(lbl.clone()).or_insert(0) += 1;
    }
    counts.into_iter().max_by_key(|(_k, v)| *v).unwrap().0
}

/// Compute the Shannon entropy of a set of labels.
fn entropy<L: Eq + std::hash::Hash>(labels: &[L]) -> f64 {
    let mut counts = HashMap::new();
    for lbl in labels {
        *counts.entry(lbl).or_insert(0) += 1;
    }
    let n = labels.len() as f64;
    let mut ent = 0.0;
    for (_lbl, count) in counts.into_iter() {
        let p = count as f64 / n;
        ent -= p * log2(p);
    }
    ent
}

/// Compute log base 2 of `x`.
fn log2(x: f64) -> f64 {
    x.ln() / std::f64::consts::LN_2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_id3_basic() {
        // Simple dataset: [Color, Shape] => {Yes/No}
        let data = vec![
            vec!["Red".to_string(), "Round".to_string()],
            vec!["Blue".to_string(), "Square".to_string()],
            vec!["Red".to_string(), "Square".to_string()],
            vec!["Blue".to_string(), "Round".to_string()],
        ];
        let labels = vec!["Yes", "No", "Yes", "No"];
        let tree = DecisionTree::fit(&data, &labels, None, DecisionTreeAlgorithm::ID3);

        // Predict an unseen pattern
        let pred = tree.predict(&["Red".to_string(), "Round".to_string()]);
        assert_eq!(pred, "Yes");
    }

    #[test]
    fn test_c45_continuous() {
        // We'll define a small dataset with continuous and discrete columns.
        // Format: [Temperature, Color], label is "Buy" or "NoBuy"
        // We'll see if a threshold on Temperature emerges.
        let data = vec![
            vec!["30.5".to_string(), "Red".to_string()],
            vec!["35.0".to_string(), "Blue".to_string()],
            vec!["40.0".to_string(), "Red".to_string()],
            vec!["45.0".to_string(), "Blue".to_string()],
            vec!["50.0".to_string(), "Blue".to_string()],
        ];
        let labels = vec!["Buy", "NoBuy", "Buy", "NoBuy", "NoBuy"];
        let tree = DecisionTree::fit(&data, &labels, None, DecisionTreeAlgorithm::C45);

        // Check some predictions
        let pred1 = tree.predict(&["38.0".to_string(), "Red".to_string()]);
        // Possibly "Buy" if the threshold is around 37.5-ish
        assert_eq!(pred1, "Buy");

        let pred2 = tree.predict(&["47.0".to_string(), "Blue".to_string()]);
        assert_eq!(pred2, "NoBuy");
    }
}
