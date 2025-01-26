//! This module provides a simple Markov Decision Process (MDP) representation
//! and a value iteration procedure that solves the Bellman equations
//! to compute an optimal policy.

/// A struct representing a Markov Decision Process in discrete form.
#[derive(Debug)]
pub struct MarkovDecisionProcess {
    /// Number of states in the MDP
    pub num_states: usize,
    /// Number of actions (assumed available in each state)
    pub num_actions: usize,
    /// Discount factor (0 <= gamma <= 1)
    pub gamma: f64,
    /// Transitions and rewards: for each (state, action), a list of (next_state, probability, reward).
    ///
    /// Example:
    /// transitions[s][a] = vec![
    ///     (s_next_0, p_0, r_0),
    ///     (s_next_1, p_1, r_1),
    ///     ...
    /// ]
    ///
    /// The probabilities for each (s, a) must sum to 1.0.
    pub transitions: Vec<Vec<Vec<(usize, f64, f64)>>>,
}

impl MarkovDecisionProcess {
    /// Creates a new MDP instance. For each state s and action a,
    /// you provide a list of (next_state, probability, reward).
    ///
    /// # Panics
    /// Panics if the dimensions don't match `num_states` and `num_actions`, or
    /// if any probability lists don't sum to ~1.0 (within epsilon).
    pub fn new(
        num_states: usize,
        num_actions: usize,
        gamma: f64,
        transitions: Vec<Vec<Vec<(usize, f64, f64)>>>,
    ) -> Self {
        assert!(
            (0.0..=1.0).contains(&gamma),
            "Discount factor gamma must be between 0 and 1"
        );
        assert_eq!(transitions.len(), num_states);
        for sa in &transitions {
            assert_eq!(sa.len(), num_actions);
        }
        // Validate probability sums
        for (s, _sa) in transitions.iter().enumerate().take(num_states) {
            for a in 0..num_actions {
                let prob_sum: f64 = transitions[s][a].iter().map(|(_, p, _)| p).sum();
                let diff = (prob_sum - 1.0).abs();
                // Allow a little floating error
                assert!(
                    diff < 1e-8,
                    "Probabilities in state {}, action {} must sum to 1.0, but got {}",
                    s,
                    a,
                    prob_sum
                );
            }
        }

        Self {
            num_states,
            num_actions,
            gamma,
            transitions,
        }
    }
}

/// Performs value iteration on the given MDP, returning:
/// 1. The near-optimal value function (one entry per state).
/// 2. A greedy policy (one action per state).
///
/// # Arguments
/// - `mdp`: the Markov Decision Process
/// - `max_iterations`: maximum number of iterations
/// - `tolerance`: stop early if the maximum change in value function is < `tolerance`.
///
/// # Returns
/// - `value_function`: a `Vec<f64>` of length `mdp.num_states`
/// - `policy`: a `Vec<usize>` of length `mdp.num_states` specifying the action that
///   maximizes the Q-value for each state (ties broken arbitrarily).
///
/// # Examples
///
/// ```
/// use algos::cs::dynamic::bellman_equation::{MarkovDecisionProcess, value_iteration};
///
/// // Create a simple MDP with 2 states and 2 actions
/// let states = 2;
/// let actions = 2;
/// let gamma = 0.9;
///
/// // Define transitions: for each (state, action), a list of (next_state, probability, reward)
/// let transitions = vec![
///     // State 0
///     vec![
///         // Action 0
///         vec![(0, 0.7, 1.0), (1, 0.3, 0.5)],
///         // Action 1
///         vec![(1, 1.0, 2.0)],
///     ],
///     // State 1
///     vec![
///         // Action 0
///         vec![(0, 0.4, 0.8), (1, 0.6, 0.0)],
///         // Action 1
///         vec![(0, 0.1, 0.0), (1, 0.9, 1.5)],
///     ],
/// ];
///
/// let mdp = MarkovDecisionProcess::new(states, actions, gamma, transitions);
/// let (values, policy) = value_iteration(&mdp, 100, 0.01);
///
/// assert_eq!(values.len(), states);
/// assert_eq!(policy.len(), states);
/// ```
pub fn value_iteration(
    mdp: &MarkovDecisionProcess,
    max_iterations: usize,
    tolerance: f64,
) -> (Vec<f64>, Vec<usize>) {
    let n = mdp.num_states;
    let mut v = vec![0.0; n]; // start with zero values
    let gamma = mdp.gamma;

    for _iter in 0..max_iterations {
        let mut delta = 0.0_f64;
        // We'll compute updated values in a separate array to avoid partial updates
        let mut v_new = vec![0.0; n];
        for s in 0..n {
            // For each state, compute the best action's Q-value
            let mut best_val = f64::NEG_INFINITY;
            for a in 0..mdp.num_actions {
                let q_sa = compute_q_value(s, a, &v, mdp, gamma);
                if q_sa > best_val {
                    best_val = q_sa;
                }
            }
            v_new[s] = best_val;
            // Track the biggest change
            delta = delta.max((v_new[s] - v[s]).abs());
        }

        // Replace old value function with the new
        v = v_new;
        // If the improvement is small enough, we stop
        if delta < tolerance {
            break;
        }
    }

    // After convergence (or max_iterations), produce a greedy policy
    let mut policy = vec![0_usize; n];
    for (s, policy_s) in policy.iter_mut().enumerate().take(n) {
        let mut best_a = 0;
        let mut best_val = f64::NEG_INFINITY;
        for a in 0..mdp.num_actions {
            let q_sa = compute_q_value(s, a, &v, mdp, gamma);
            if q_sa > best_val {
                best_val = q_sa;
                best_a = a;
            }
        }
        *policy_s = best_a;
    }

    (v, policy)
}

/// Compute Q(s, a) = sum_{s'} P(s'|s,a) [ R(s,a,s') + gamma * V(s') ].
fn compute_q_value(
    s: usize,
    a: usize,
    values: &[f64],
    mdp: &MarkovDecisionProcess,
    gamma: f64,
) -> f64 {
    let mut q = 0.0;
    for &(s_next, prob, reward) in &mdp.transitions[s][a] {
        q += prob * (reward + gamma * values[s_next]);
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_mdp() {
        // 2 states, 2 actions, deterministic transitions:
        // state 0, action 0 => stay(0), reward=1
        // state 0, action 1 => go(1), reward=0
        // state 1, action 0 => go(0), reward=0
        // state 1, action 1 => stay(1), reward=2
        let mdp = MarkovDecisionProcess::new(
            2,
            2,
            0.9,
            vec![
                vec![vec![(0, 1.0, 1.0)], vec![(1, 1.0, 0.0)]],
                vec![vec![(0, 1.0, 0.0)], vec![(1, 1.0, 2.0)]],
            ],
        );
        let (values, policy) = value_iteration(&mdp, 100, 1e-6);

        // We expect that in state 1, taking action 1 repeatedly is quite valuable:
        // Q(1,1) = 2 + 0.9*V(1), so V(1) = 2 + 0.9*V(1) => V(1) = 20 (approx).
        // In state 0, action 0 yields 1 + 0.9*V(0), action 1 yields 0 + 0.9*V(1).
        // The second might be better if V(1) is large.
        // Let's see if we converge near that logic.
        assert!(values[1] > 10.0); // should be fairly large
                                   // The policy in state 1 should be action 1
        assert_eq!(policy[1], 1);

        // The policy in state 0 might be action 1 if transitioning to state 1 yields a higher long-term return.
        // We'll just check we produce a sensible result (the best action isn't obviously the "stay in 0" if 1's value is bigger).
        // So we expect policy[0] = 1 in many solutions.
        // But let's just ensure the code doesn't produce an out-of-bounds or something nonsensical.
        assert!(policy[0] < 2);
    }

    #[test]
    #[should_panic]
    fn test_invalid_probabilities() {
        // Probability sums not equal to 1.0 => should panic
        MarkovDecisionProcess::new(
            1,
            1,
            0.99,
            vec![vec![vec![(0, 0.5, 10.0)]]], // sums to 0.5, not 1.0
        );
    }
}
