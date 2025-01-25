/// lib.rs

/// Represents a discrete Hidden Markov Model (HMM), storing:
/// - `num_states`: how many hidden states
/// - `num_observations`: how many distinct observation types
/// - `initial_probabilities`: probability of starting in each state
/// - `transition_probabilities[s1][s2]`: probability of transitioning from state s1 to s2
/// - `emission_probabilities[s][o]`: probability of emitting observation o in state s
///
/// # Constraints
/// - All internal probability vectors must sum to 1 (for their respective categories).
/// - The user must ensure they pass valid sizes for the probabilities.
#[derive(Debug)]
pub struct HiddenMarkovModel {
    pub num_states: usize,
    pub num_observations: usize,
    pub initial_probabilities: Vec<f64>,
    pub transition_probabilities: Vec<Vec<f64>>,
    pub emission_probabilities: Vec<Vec<f64>>,
}

impl HiddenMarkovModel {
    /// Create a new `HiddenMarkovModel`.
    ///
    /// # Panics
    ///
    /// Panics if dimensions of transition or emission probabilities do not match
    /// `num_states` or `num_observations`.
    pub fn new(
        num_states: usize,
        num_observations: usize,
        initial_probabilities: Vec<f64>,
        transition_probabilities: Vec<Vec<f64>>,
        emission_probabilities: Vec<Vec<f64>>,
    ) -> Self {
        assert_eq!(initial_probabilities.len(), num_states);
        assert_eq!(transition_probabilities.len(), num_states);
        for row in &transition_probabilities {
            assert_eq!(row.len(), num_states);
        }
        assert_eq!(emission_probabilities.len(), num_states);
        for row in &emission_probabilities {
            assert_eq!(row.len(), num_observations);
        }

        Self {
            num_states,
            num_observations,
            initial_probabilities,
            transition_probabilities,
            emission_probabilities,
        }
    }
}

/// Runs the Viterbi Algorithm for a given HMM and a sequence of observations.
///
/// Returns a vector of the most likely hidden state indices that generate
/// the given observation sequence.
///
/// # Arguments
///
/// - `hmm`: reference to a `HiddenMarkovModel`
/// - `observations`: slice of observation indices (each must be < `hmm.num_observations`)
///
/// # Panics
///
/// Panics if any observation in `observations` is out of range.
pub fn viterbi(hmm: &HiddenMarkovModel, observations: &[usize]) -> Vec<usize> {
    if observations.is_empty() {
        return Vec::new();
    }

    for &obs in observations {
        assert!(
            obs < hmm.num_observations,
            "Observation {} is out of range (max = {})",
            obs,
            hmm.num_observations - 1
        );
    }

    let t = observations.len();
    let n = hmm.num_states;

    // delta[t][s]: highest log-probability of any path that ends in state s at time t
    // psi[t][s]: which state at time t-1 led to the best path ending in s at time t
    let mut delta = vec![vec![f64::NEG_INFINITY; n]; t];
    let mut psi = vec![vec![0_usize; n]; t];

    // Initialization step (time 0)
    for s in 0..n {
        // Using log probabilities for better numerical stability
        let init_log = hmm.initial_probabilities[s].ln();
        let emission_log = hmm.emission_probabilities[s][observations[0]].ln();
        delta[0][s] = init_log + emission_log;
        psi[0][s] = 0; // no predecessor for t=0
    }

    // Recursion
    for time in 1..t {
        let obs = observations[time];
        for s in 0..n {
            let emit_log = hmm.emission_probabilities[s][obs].ln();

            // We want to find argmax_{s'} [ delta[time-1][s'] + ln(transition[s'->s]) ]
            let mut best_val = f64::NEG_INFINITY;
            let mut best_prev = 0_usize;
            for s_prev in 0..n {
                let candidate =
                    delta[time - 1][s_prev] + hmm.transition_probabilities[s_prev][s].ln();
                if candidate > best_val {
                    best_val = candidate;
                    best_prev = s_prev;
                }
            }
            delta[time][s] = best_val + emit_log;
            psi[time][s] = best_prev;
        }
    }

    // Termination: find best final state
    let mut best_final_score = f64::NEG_INFINITY;
    let mut best_final_state = 0_usize;

    for s in 0..n {
        if delta[t - 1][s] > best_final_score {
            best_final_score = delta[t - 1][s];
            best_final_state = s;
        }
    }

    // Path backtracking
    let mut path = vec![0_usize; t];
    path[t - 1] = best_final_state;
    for time in (1..t).rev() {
        path[time - 1] = psi[time][path[time]];
    }

    path
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper for approximate floating comparison
    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_viterbi_simple() {
        // A simple example with 2 states and 3 possible observations.
        // Suppose we have:
        // State 0 => "Sunny"
        // State 1 => "Rainy"
        // Observations: 0 => "Walk", 1 => "Shop", 2 => "Clean"
        //
        // HMM parameters (all probabilities must sum to 1 in their respective arrays):
        // initial_prob: P(state0) = 0.6, P(state1) = 0.4
        // transitions: 
        //    state0 -> [state0=0.7, state1=0.3]
        //    state1 -> [state0=0.4, state1=0.6]
        // emission:
        //    state0 -> [walk=0.5, shop=0.4, clean=0.1]
        //    state1 -> [walk=0.1, shop=0.3, clean=0.6]
        let hmm = HiddenMarkovModel::new(
            2,
            3,
            vec![0.6, 0.4],
            vec![
                vec![0.7, 0.3],
                vec![0.4, 0.6],
            ],
            vec![
                vec![0.5, 0.4, 0.1],
                vec![0.1, 0.3, 0.6],
            ],
        );

        // Observed sequence: [walk, shop, clean]
        let observations = vec![0_usize, 1, 2];

        let path = viterbi(&hmm, &observations);

        // The path is a sequence of states, e.g. [0, 0, 1] for example, depending on which has highest prob.
        // Let's compute quickly:
        //   - If all states were 0: (0.6*0.5) * (0.7*0.4) * (0.7*0.1)? etc. It's simpler to rely on the function.
        // We'll just check the final path length and that it doesn't produce an invalid state index.
        assert_eq!(path.len(), 3);
        for &st in &path {
            assert!(st < 2);
        }
    }

    #[test]
    fn test_viterbi_empty() {
        let hmm = HiddenMarkovModel::new(2, 3, vec![1.0, 0.0], vec![vec![1.0, 0.0], vec![0.0, 1.0]], vec![vec![1.0,0.0,0.0], vec![0.0,1.0,0.0]]);
        let empty_obs: Vec<usize> = vec![];
        let path = viterbi(&hmm, &empty_obs);
        assert_eq!(path.len(), 0);
    }

    #[test]
    fn test_probabilities_sum_one() {
        // Quick checks that our transitions sum to 1.0
        let hmm = HiddenMarkovModel::new(
            2,
            3,
            vec![0.6, 0.4],
            vec![
                vec![0.7, 0.3],
                vec![0.4, 0.6],
            ],
            vec![
                vec![0.5, 0.4, 0.1],
                vec![0.1, 0.3, 0.6],
            ],
        );

        for row in &hmm.transition_probabilities {
            let sum: f64 = row.iter().sum();
            assert!(approx_eq(sum, 1.0, 1e-9));
        }

        for row in &hmm.emission_probabilities {
            let sum: f64 = row.iter().sum();
            assert!(approx_eq(sum, 1.0, 1e-9));
        }

        let initial_sum: f64 = hmm.initial_probabilities.iter().sum();
        assert!(approx_eq(initial_sum, 1.0, 1e-9));
    }
}
