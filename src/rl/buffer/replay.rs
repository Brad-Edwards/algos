use rand::seq::SliceRandom;
use std::collections::VecDeque;

use super::Buffer;
use crate::rl::env::Transition;

/// Standard uniform replay buffer for off-policy algorithms (DQN, etc.).
pub struct ReplayBuffer {
    storage: VecDeque<Transition>,
    max_capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            storage: VecDeque::with_capacity(capacity),
            max_capacity: capacity,
        }
    }
}

impl Buffer for ReplayBuffer {
    fn push(&mut self, transition: Transition) {
        if self.storage.len() >= self.max_capacity {
            self.storage.pop_front();
        }
        self.storage.push_back(transition);
    }

    fn sample(&self, n: usize) -> Vec<Transition> {
        let mut rng = rand::thread_rng();
        let indices: Vec<usize> = (0..self.storage.len()).collect();
        indices
            .choose_multiple(&mut rng, n.min(self.storage.len()))
            .map(|&i| self.storage[i].clone())
            .collect()
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
    fn test_push_and_len() {
        let mut buf = ReplayBuffer::new(10);
        assert!(buf.is_empty());
        buf.push(make_transition(1.0));
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn test_capacity_eviction() {
        let mut buf = ReplayBuffer::new(3);
        for i in 0..5 {
            buf.push(make_transition(i as f64));
        }
        assert_eq!(buf.len(), 3);
        // Oldest transitions should be evicted
        assert_eq!(buf.storage[0].reward, 2.0);
    }

    #[test]
    fn test_sample() {
        let mut buf = ReplayBuffer::new(100);
        for i in 0..50 {
            buf.push(make_transition(i as f64));
        }
        let batch = buf.sample(10);
        assert_eq!(batch.len(), 10);
    }

    #[test]
    fn test_sample_more_than_available() {
        let mut buf = ReplayBuffer::new(100);
        buf.push(make_transition(1.0));
        let batch = buf.sample(10);
        assert_eq!(batch.len(), 1);
    }
}
