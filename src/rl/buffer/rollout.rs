use super::Buffer;
use crate::rl::env::Transition;

/// Rollout buffer for on-policy algorithms (PPO, TRPO, A2C).
///
/// Unlike a replay buffer, the rollout buffer is consumed entirely after
/// each policy update and then cleared.
pub struct RolloutBuffer {
    storage: Vec<Transition>,
    max_capacity: usize,
}

impl RolloutBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            storage: Vec::with_capacity(capacity),
            max_capacity: capacity,
        }
    }

    /// Drain all stored transitions (consuming the buffer contents).
    pub fn drain(&mut self) -> Vec<Transition> {
        self.storage.drain(..).collect()
    }
}

impl Buffer for RolloutBuffer {
    fn push(&mut self, transition: Transition) {
        if self.storage.len() < self.max_capacity {
            self.storage.push(transition);
        }
    }

    fn sample(&self, n: usize) -> Vec<Transition> {
        self.storage.iter().take(n).cloned().collect()
    }

    fn len(&self) -> usize {
        self.storage.len()
    }

    fn capacity(&self) -> usize {
        self.max_capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_transition(reward: f64) -> Transition {
        Transition {
            state: vec![0.0],
            action: 0,
            reward,
            next_state: vec![1.0],
            done: false,
        }
    }

    #[test]
    fn test_drain() {
        let mut buf = RolloutBuffer::new(100);
        buf.push(make_transition(1.0));
        buf.push(make_transition(2.0));
        let data = buf.drain();
        assert_eq!(data.len(), 2);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_capacity_limit() {
        let mut buf = RolloutBuffer::new(2);
        buf.push(make_transition(1.0));
        buf.push(make_transition(2.0));
        buf.push(make_transition(3.0));
        assert_eq!(buf.len(), 2);
    }
}
