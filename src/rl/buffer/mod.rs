mod replay;
mod rollout;

pub use replay::ReplayBuffer;
pub use rollout::RolloutBuffer;

use crate::rl::env::Transition;

/// Trait for experience storage backends.
///
/// Replay buffers store transitions for off-policy learning. On-policy
/// algorithms use `RolloutBuffer` instead. Both PER and HER can be
/// implemented as drop-in replacements via this trait.
pub trait Buffer {
    /// Store a transition.
    fn push(&mut self, transition: Transition);

    /// Sample a batch of transitions.
    fn sample(&self, n: usize) -> Vec<Transition>;

    /// Number of stored transitions.
    fn len(&self) -> usize;

    /// Whether the buffer is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Maximum capacity.
    fn capacity(&self) -> usize;
}
