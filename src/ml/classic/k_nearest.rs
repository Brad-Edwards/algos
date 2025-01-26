/// lib.rs
use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;

/// A simple k-NN classifier that stores training data and performs majority-vote classification.
///
/// # Type Parameters
/// - `L`: The label type. Must be `Eq + Hash` so we can store counts in a `HashMap`.
///
/// # Fields
/// - `k`: number of neighbors to consider.
/// - `features`: a vector of training feature vectors (each is a Vec<f64>).
/// - `labels`: corresponding labels for each feature vector.
/// - `_label_marker`: a zero-sized marker to ensure the struct is typed by the label.
#[derive(Debug, Clone)]
pub struct KNNClassifier<L: Eq + Hash + Clone> {
    pub k: usize,
    pub features: Vec<Vec<f64>>,
    pub labels: Vec<L>,
    _label_marker: PhantomData<L>,
}

impl<L: Eq + Hash + Clone> KNNClassifier<L> {
    /// Constructs a new `KNNClassifier`.
    ///
    /// # Panics
    ///
    /// - If `k == 0`.
    /// - If `features.len() != labels.len()`.
    /// - If any feature vector is empty.
    pub fn new(k: usize, features: Vec<Vec<f64>>, labels: Vec<L>) -> Self {
        assert!(k > 0, "k must be > 0");
        let n = features.len();
        assert_eq!(n, labels.len(), "features and labels must have same length");
        for f in &features {
            assert!(
                !f.is_empty(),
                "All feature vectors must have at least one dimension"
            );
        }

        Self {
            k,
            features,
            labels,
            _label_marker: PhantomData,
        }
    }

    /// Predict the label for a single query point using majority vote among its `k` nearest neighbors.
    ///
    /// # Panics
    ///
    /// - If the classifier has no training data.
    ///
    /// # Example
    ///
    /// ```
    /// use knn::{KNNClassifier};
    ///
    /// let features = vec![
    ///     vec![1.0, 2.0],
    ///     vec![2.0, 3.0],
    ///     vec![3.0, 3.0],
    ///     vec![6.0, 7.0],
    /// ];
    /// let labels = vec!["A", "A", "B", "B"];
    ///
    /// let knn = KNNClassifier::new(3, features, labels);
    ///
    /// // Predict for a new point
    /// let test_point = vec![2.1, 2.9];
    /// let predicted_label = knn.predict(&test_point);
    /// println!("Predicted label: {}", predicted_label);
    /// ```
    pub fn predict(&self, point: &[f64]) -> L {
        assert!(
            !self.features.is_empty(),
            "No training data in the classifier"
        );
        let neighbors = self.find_k_nearest(point);
        // Majority vote among these neighbors
        self.majority_vote(neighbors)
    }

    /// Predict labels for multiple query points at once.
    pub fn predict_batch(&self, points: &[Vec<f64>]) -> Vec<L> {
        points.iter().map(|p| self.predict(p)).collect()
    }

    /// Find the indices of the k nearest neighbors of `point`.
    fn find_k_nearest(&self, point: &[f64]) -> Vec<usize> {
        // We will store (distance, index), then sort by distance
        let mut dists: Vec<(f64, usize)> = self
            .features
            .iter()
            .enumerate()
            .map(|(i, f)| (euclidean_distance_sq(f, point), i))
            .collect();

        // Partial sort or nth_element approach could be used for efficiency, but
        // here we simply sort by ascending distance.
        dists.sort_by(|(d1, _), (d2, _)| d1.partial_cmp(d2).unwrap());

        // Extract indices of the first k
        dists.iter().take(self.k).map(|&(_, i)| i).collect()
    }

    /// Majority vote over the labels of neighbors. If there's a tie, the label with the largest
    /// count encountered first in iteration order is returned (this is arbitrary tie-breaking).
    fn majority_vote(&self, neighbor_indices: Vec<usize>) -> L {
        let mut counts = HashMap::<L, usize>::new();
        for idx in neighbor_indices {
            let label = &self.labels[idx];
            *counts.entry(label.clone()).or_insert(0) += 1;
        }
        // Find label with max count
        counts
            .into_iter()
            .max_by_key(|(_label, count)| *count)
            .unwrap()
            .0
    }
}

/// Returns the **squared** Euclidean distance between two vectors.
/// Using squared distance to avoid unnecessary sqrt in comparisons, but final
/// result is consistent with actual distance for nearest-neighbor queries.
fn euclidean_distance_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_knn() {
        // Construct a small dataset
        let features = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![2.5, 2.7],
            vec![10.0, 10.0],
        ];
        let labels = vec!["A", "A", "B", "B"];

        // Build KNN with k=3
        let knn = KNNClassifier::new(3, features, labels);

        // Predict a label for point near the first cluster
        let pred1 = knn.predict(&[2.1, 2.9]);
        // Should be "A" or "B" but likely "A" since 2 of the 3 nearest neighbors are A
        assert_eq!(pred1, "A");

        // Predict for a point near [10, 10]
        let pred2 = knn.predict(&[9.5, 9.7]);
        assert_eq!(pred2, "B");
    }

    #[test]
    fn test_empty_features_panic() {
        // Expect panic if no features
        let features: Vec<Vec<f64>> = vec![];
        let labels: Vec<&str> = vec![];
        let result = std::panic::catch_unwind(|| {
            KNNClassifier::new(3, features, labels);
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_predict_no_data_panic() {
        let knn = KNNClassifier::<&str> {
            k: 3,
            features: vec![],
            labels: vec![],
            _label_marker: PhantomData,
        };
        let result = std::panic::catch_unwind(|| {
            knn.predict(&[1.0, 2.0]);
        });
        assert!(result.is_err());
    }
}
