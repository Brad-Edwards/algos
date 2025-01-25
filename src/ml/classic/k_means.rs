/// lib.rs
use rand::seq::SliceRandom;
use rand::thread_rng;

/// Configuration options for k-means clustering.
#[derive(Debug, Clone)]
pub struct KMeansConfig {
    /// Number of clusters to find.
    pub k: usize,
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence tolerance. If the movement of all centroids is below this,
    /// the algorithm stops early.
    pub tolerance: f64,
}

impl KMeansConfig {
    /// Create a new config with default values for max_iterations (300) and tolerance (1e-4).
    pub fn new(k: usize) -> Self {
        Self {
            k,
            max_iterations: 300,
            tolerance: 1e-4,
        }
    }

    /// Customize the maximum number of iterations.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Customize the convergence tolerance.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }
}

/// Runs k-means clustering on the provided dataset, returning:
/// 1) A vector of assignments (cluster indices) for each data point,
/// 2) A vector of final centroids (each centroid is a Vec<f64>).
///
/// # Arguments
///
/// - `data`: A slice of data points. Each data point is itself a slice (`&[f64]`).
/// - `config`: KMeansConfig specifying the number of clusters (k), etc.
///
/// # Panics
///
/// - If `config.k` is 0 or greater than the number of data points.
/// - If any data point has length 0 (empty feature vector).
/// - If the dataset is empty.
///
/// # Example
///
/// ```
/// use kmeans::{KMeansConfig, kmeans};
///
/// let data = vec![
///     vec![1.0, 2.0],
///     vec![1.5, 1.8],
///     vec![5.0, 8.0],
///     vec![8.0, 8.0],
/// ];
///
/// let config = KMeansConfig::new(2);
/// let (assignments, centroids) = kmeans(&data, &config);
///
/// println!("Assignments: {:?}", assignments);
/// println!("Centroids: {:?}", centroids);
/// ```
pub fn kmeans(data: &[Vec<f64>], config: &KMeansConfig) -> (Vec<usize>, Vec<Vec<f64>>) {
    // Basic checks
    if data.is_empty() {
        panic!("Empty dataset provided.");
    }
    let n = data.len();
    let dim = data[0].len();
    if dim == 0 {
        panic!("Data points must have at least one dimension.");
    }
    if config.k == 0 || config.k > n {
        panic!("Invalid number of clusters k = {} for dataset of size {}", config.k, n);
    }

    // Initialize centroids by sampling k distinct points
    let mut rng = thread_rng();
    let mut centroids: Vec<Vec<f64>> = data
        .choose_multiple(&mut rng, config.k)
        .cloned()
        .collect();

    // Vector of cluster assignments for each point
    let mut assignments = vec![0_usize; n];

    // k-means main loop
    for _iter in 0..config.max_iterations {
        let mut changed = false;

        // 1. Assignment step: assign each point to the nearest centroid
        for (i, point) in data.iter().enumerate() {
            let mut best_cluster = assignments[i];
            let mut best_dist = distance_sq(point, &centroids[best_cluster]);
            for cluster_idx in 0..config.k {
                let dist = distance_sq(point, &centroids[cluster_idx]);
                if dist < best_dist {
                    best_dist = dist;
                    best_cluster = cluster_idx;
                }
            }
            if best_cluster != assignments[i] {
                assignments[i] = best_cluster;
                changed = true;
            }
        }

        // 2. Update step: recompute centroids based on the new assignments
        // We'll accumulate sums in each cluster and then divide by counts
        let mut sums = vec![vec![0.0; dim]; config.k];
        let mut counts = vec![0_usize; config.k];
        for (i, point) in data.iter().enumerate() {
            let c = assignments[i];
            counts[c] += 1;
            for d in 0..dim {
                sums[c][d] += point[d];
            }
        }

        let mut max_centroid_shift_sq = 0.0;
        for cluster_idx in 0..config.k {
            if counts[cluster_idx] > 0 {
                let mut new_centroid = vec![0.0; dim];
                for d in 0..dim {
                    new_centroid[d] = sums[cluster_idx][d] / counts[cluster_idx] as f64;
                }
                // Compute movement (euclidean) of the centroid
                let shift_sq = distance_sq(&centroids[cluster_idx], &new_centroid);
                if shift_sq > max_centroid_shift_sq {
                    max_centroid_shift_sq = shift_sq;
                }
                centroids[cluster_idx] = new_centroid;
            }
            // if a cluster gets zero points, we leave its centroid as is 
            // or could re-initialize randomly, but standard practice is to keep it.
        }

        // Check if we've converged sufficiently (no assignment changes or centroid shift < tolerance)
        if !changed || max_centroid_shift_sq < config.tolerance * config.tolerance {
            break;
        }
    }

    (assignments, centroids)
}

/// Compute the squared Euclidean distance between two points of the same dimension.
/// Using squared distance to avoid unnecessary sqrt computations during comparisons.
fn distance_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .fold(0.0, |acc, (&x, &y)| acc + (x - y).powi(2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn test_empty_data() {
        let config = KMeansConfig::new(3);
        let data: Vec<Vec<f64>> = vec![];
        let _ = kmeans(&data, &config);
    }

    #[test]
    #[should_panic]
    fn test_invalid_k() {
        let config = KMeansConfig::new(5);
        let data = vec![vec![1.0, 2.0], vec![2.0, 3.0]];
        let _ = kmeans(&data, &config);
    }

    #[test]
    fn test_basic_run() {
        // 4 points in 2D, 2 clusters
        let data = vec![
            vec![1.0, 2.0],
            vec![1.5, 1.8],
            vec![5.0, 8.0],
            vec![8.0, 8.0],
        ];
        let config = KMeansConfig::new(2).with_max_iterations(50).with_tolerance(1e-4);
        let (assignments, centroids) = kmeans(&data, &config);

        // We won't test the exact assignment or centroids, but ensure we have valid output.
        assert_eq!(assignments.len(), data.len());
        assert_eq!(centroids.len(), 2);
        for c in &centroids {
            assert_eq!(c.len(), 2);
        }
    }
}
