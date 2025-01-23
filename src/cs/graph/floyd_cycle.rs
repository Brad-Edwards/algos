//! # Floyd's Cycle Detection (Tortoise and Hare)
//!
//! This module provides a standard implementation of Floyd's
//! Cycle Detection (a.k.a. Tortoise and Hare) for singly-linked lists. This
//! algorithm detects whether a cycle exists in \( O(n) \) time and \( O(1) \)
//! extra space, and can also identify the node where the cycle begins.
//!
//! ## Overview
//!
//! Floyd's Cycle Detection uses two pointers (slow and fast). Slow advances by
//! one node at a time, while fast advances by two nodes at a time. If they ever
//! point to the same node, a cycle exists. To find the *start* of the cycle,
//! reset one pointer to the head and advance both by one node at a time. The
//! node where they meet is the start of the cycle.
//!
//! ## Example Usage
//!
//! ```rust
//! use std::rc::Rc;
//! use std::cell::RefCell;
//!
//! // Suppose we have the following list: 1 -> 2 -> 3 -> 4 -> 5
//! // We'll create it and introduce a cycle from 5 back to node 3.
//! use floyd_cycle_detection::{ListNode, has_cycle, find_cycle_start};
//!
//! // Create each node (wrapped in Rc<RefCell<>> to allow shared ownership)
//! let node1 = Rc::new(RefCell::new(ListNode::new(1)));
//! let node2 = Rc::new(RefCell::new(ListNode::new(2)));
//! let node3 = Rc::new(RefCell::new(ListNode::new(3)));
//! let node4 = Rc::new(RefCell::new(ListNode::new(4)));
//! let node5 = Rc::new(RefCell::new(ListNode::new(5)));
//!
//! // Link them: 1->2->3->4->5
//! node1.borrow_mut().next = Some(node2.clone());
//! node2.borrow_mut().next = Some(node3.clone());
//! node3.borrow_mut().next = Some(node4.clone());
//! node4.borrow_mut().next = Some(node5.clone());
//!
//! // Introduce a cycle: 5 -> 3
//! node5.borrow_mut().next = Some(node3.clone());
//!
//! let head = Some(node1.clone());
//!
//! assert_eq!(has_cycle(&head), true);
//! let start_node = find_cycle_start(&head).unwrap();
//! assert_eq!(start_node.borrow().val, 3); // The cycle starts at node with value 3
//! ```

use std::cell::RefCell;
use std::rc::Rc;

/// A singly-linked list node that can share ownership (via `Rc`) and be
/// modified (via `RefCell`). This allows creating cycles for testing or
/// demonstration of cycle detection.
#[derive(Debug)]
pub struct ListNode<T> {
    pub val: T,
    pub next: Option<Rc<RefCell<ListNode<T>>>>,
}

impl<T> ListNode<T> {
    /// Creates a new `ListNode` with the given value and no next pointer.
    pub fn new(val: T) -> Self {
        ListNode { val, next: None }
    }
}

/// Determines if a singly-linked list has a cycle using Floyd's Tortoise and Hare.
/// - `head`: The head node of the list (or `None` if empty).
/// - Returns `true` if there's a cycle, `false` otherwise.
pub fn has_cycle<T>(head: &Option<Rc<RefCell<ListNode<T>>>>) -> bool {
    // If list is empty or has no next, no cycle
    let mut slow = head.clone();
    let mut fast = head.clone();

    while let Some(f) = fast {
        // Advance fast pointer by one
        let next_fast = f.borrow().next.clone();
        if let Some(f2) = next_fast {
            // Advance fast pointer by second step
            fast = f2.borrow().next.clone();
        } else {
            // Next step not available => no cycle
            return false;
        }

        // Advance slow pointer by one
        if let Some(s) = slow.clone() {
            slow = s.borrow().next.clone();
        }

        // If they meet, cycle detected
        if let (Some(sf), Some(ff)) = (slow.clone(), fast.clone()) {
            if Rc::ptr_eq(&sf, &ff) {
                return true;
            }
        } else {
            return false;
        }
    }
    false
}

/// If a cycle exists, returns the node (as `Rc<RefCell<ListNode<T>>>`) where
/// the cycle begins. If no cycle exists, returns `None`.
///
/// The algorithm first uses Floyd's Tortoise and Hare to detect a meeting point.
/// If no meeting point exists, there's no cycle. If it does exist, we reset one
/// pointer to head and advance both one step at a time until they meet again.
/// That node is the start of the cycle.
pub fn find_cycle_start<T>(
    head: &Option<Rc<RefCell<ListNode<T>>>>,
) -> Option<Rc<RefCell<ListNode<T>>>> {
    // Early exit for empty list
    if head.is_none() {
        return None;
    }
    let mut slow = head.clone();
    let mut fast = head.clone();
    let mut intersection: Option<Rc<RefCell<ListNode<T>>>> = None;

    // Phase 1: Detect cycle
    while let Some(f) = fast {
        let next_fast = f.borrow().next.clone();
        if let Some(f2) = next_fast {
            fast = f2.borrow().next.clone();
        } else {
            return None; // No cycle if we can't advance fast pointer
        }

        if let Some(s) = slow.clone() {
            slow = s.borrow().next.clone();
        }

        if let (Some(sf), Some(ff)) = (slow.clone(), fast.clone()) {
            if Rc::ptr_eq(&sf, &ff) {
                intersection = Some(sf);
                break;
            }
        } else {
            return None;
        }
    }

    // If no intersection was found, no cycle
    if intersection.is_none() {
        return None;
    }

    // Phase 2: Find start of cycle
    let mut ptr1 = head.clone();
    let mut ptr2 = intersection;
    while let (Some(p1), Some(p2)) = (ptr1, ptr2) {
        if Rc::ptr_eq(&p1, &p2) {
            return Some(p1);
        }
        ptr1 = p1.borrow().next.clone();
        ptr2 = p2.borrow().next.clone();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_list() {
        assert_eq!(has_cycle::<i32>(&None), false);
        assert!(find_cycle_start::<i32>(&None).is_none());
    }

    #[test]
    fn test_single_node_no_cycle() {
        let n1 = Rc::new(RefCell::new(ListNode::new(42)));
        let head = Some(n1.clone());
        assert_eq!(has_cycle(&head), false);
        assert!(find_cycle_start(&head).is_none());
    }

    #[test]
    fn test_two_nodes_no_cycle() {
        let n1 = Rc::new(RefCell::new(ListNode::new(1)));
        let n2 = Rc::new(RefCell::new(ListNode::new(2)));
        n1.borrow_mut().next = Some(n2.clone());

        let head = Some(n1.clone());
        assert_eq!(has_cycle(&head), false);
        assert!(find_cycle_start(&head).is_none());
    }

    #[test]
    fn test_small_cycle() {
        let n1 = Rc::new(RefCell::new(ListNode::new(1)));
        let n2 = Rc::new(RefCell::new(ListNode::new(2)));
        let n3 = Rc::new(RefCell::new(ListNode::new(3)));

        // 1 -> 2 -> 3 -> back to 2
        n1.borrow_mut().next = Some(n2.clone());
        n2.borrow_mut().next = Some(n3.clone());
        n3.borrow_mut().next = Some(n2.clone());

        let head = Some(n1.clone());
        assert_eq!(has_cycle(&head), true);

        let start = find_cycle_start(&head).unwrap();
        assert!(Rc::ptr_eq(&start, &n2));
    }

    #[test]
    fn test_longer_cycle() {
        // 1 -> 2 -> 3 -> 4 -> 5
        //              ^---------|
        let n1 = Rc::new(RefCell::new(ListNode::new(1)));
        let n2 = Rc::new(RefCell::new(ListNode::new(2)));
        let n3 = Rc::new(RefCell::new(ListNode::new(3)));
        let n4 = Rc::new(RefCell::new(ListNode::new(4)));
        let n5 = Rc::new(RefCell::new(ListNode::new(5)));

        n1.borrow_mut().next = Some(n2.clone());
        n2.borrow_mut().next = Some(n3.clone());
        n3.borrow_mut().next = Some(n4.clone());
        n4.borrow_mut().next = Some(n5.clone());

        // create cycle
        n5.borrow_mut().next = Some(n3.clone());

        let head = Some(n1.clone());

        assert_eq!(has_cycle(&head), true);
        let start = find_cycle_start(&head).unwrap();
        assert!(Rc::ptr_eq(&start, &n3));
    }

    #[test]
    fn test_no_cycle_long_list() {
        // 1 -> 2 -> 3 -> 4 -> 5 -> None
        let n1 = Rc::new(RefCell::new(ListNode::new(1)));
        let n2 = Rc::new(RefCell::new(ListNode::new(2)));
        let n3 = Rc::new(RefCell::new(ListNode::new(3)));
        let n4 = Rc::new(RefCell::new(ListNode::new(4)));
        let n5 = Rc::new(RefCell::new(ListNode::new(5)));

        n1.borrow_mut().next = Some(n2.clone());
        n2.borrow_mut().next = Some(n3.clone());
        n3.borrow_mut().next = Some(n4.clone());
        n4.borrow_mut().next = Some(n5.clone());

        let head = Some(n1.clone());
        assert_eq!(has_cycle(&head), false);
        assert!(find_cycle_start(&head).is_none());
    }
}
