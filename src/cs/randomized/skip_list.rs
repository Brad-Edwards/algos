use rand::Rng;
use std::cell::RefCell;
use std::rc::Rc;

type Link<T> = Option<Rc<RefCell<SkipNode<T>>>>;

#[derive(Debug)]
pub struct SkipNode<T> {
    pub value: Option<T>,
    pub forward: Vec<Link<T>>,
}

#[derive(Debug)]
pub struct SkipList<T> {
    head: Rc<RefCell<SkipNode<T>>>,
    pub max_level: usize,
    pub p: f64,
    pub current_level: usize,
}

impl<T: Ord> SkipList<T> {
    /// Creates a new empty skip list with a specified maximum level and probability for level promotion.
    pub fn new(max_level: usize, p: f64) -> Self {
        let head = Rc::new(RefCell::new(SkipNode {
            value: None,
            forward: vec![None; max_level],
        }));
        SkipList {
            head,
            max_level,
            p,
            current_level: 1,
        }
    }
    
    /// Randomly determines a level for a new node based on probability p.
    fn random_level(&self) -> usize {
        let mut lvl = 1;
        let mut rng = rand::thread_rng();
        while rng.gen::<f64>() < self.p && lvl < self.max_level {
            lvl += 1;
        }
        lvl
    }
    
    /// Inserts a value into the skip list.
    pub fn insert(&mut self, value: T) {
        let mut update: Vec<Rc<RefCell<SkipNode<T>>>> = vec![self.head.clone(); self.max_level];
        let mut current = self.head.clone();
        // Traverse levels from top to bottom.
        for i in (0..self.current_level).rev() {
            loop {
                let forward = current.borrow().forward[i].clone();
                match forward {
                    Some(next) => {
                        if next.borrow().value.as_ref().unwrap() < &value {
                            current = next;
                        } else {
                            break;
                        }
                    }
                    None => break,
                }
            }
            update[i] = current.clone();
        }
        // Determine node level.
        let lvl = self.random_level();
        if lvl > self.current_level {
            for i in self.current_level..lvl {
                update[i] = self.head.clone();
            }
            self.current_level = lvl;
        }
        let new_node = Rc::new(RefCell::new(SkipNode {
            value: Some(value),
            forward: vec![None; lvl],
        }));
        // Splice the new node into the list.
        for i in 0..lvl {
            let next = update[i].borrow().forward[i].clone();
            new_node.borrow_mut().forward[i] = next;
            update[i].borrow_mut().forward[i] = Some(new_node.clone());
        }
    }
    
    /// Searches for a value in the skip list. Returns true if the value is present.
    pub fn search(&self, value: &T) -> bool {
        let mut current = self.head.clone();
        for i in (0..self.current_level).rev() {
            loop {
                let forward = current.borrow().forward[i].clone();
                match forward {
                    Some(next) => {
                        if next.borrow().value.as_ref().unwrap() < value {
                            current = next;
                        } else {
                            break;
                        }
                    }
                    None => break,
                }
            }
        }
        let forward = current.borrow().forward[0].clone();
        if let Some(next) = forward {
            next.borrow().value.as_ref().unwrap() == value
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_skip_list_insert_search() {
        let mut skip_list = SkipList::new(16, 0.5);
        skip_list.insert(10);
        skip_list.insert(20);
        skip_list.insert(15);
        skip_list.insert(5);
        
        assert!(skip_list.search(&10));
        assert!(skip_list.search(&15));
        assert!(skip_list.search(&20));
        assert!(skip_list.search(&5));
        assert!(!skip_list.search(&25));
    }
}
