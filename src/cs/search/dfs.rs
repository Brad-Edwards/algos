use crate::cs::error::{Error, Result};
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

/// Represents a graph using an adjacency list.
#[derive(Debug, Clone)]
pub struct Graph<T> {
    /// Adjacency list representation of the graph
    edges: HashMap<T, Vec<T>>,
}

impl<T> Default for Graph<T>
where
    T: Eq + Hash + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Graph<T>
where
    T: Eq + Hash + Clone,
{
    /// Creates a new empty graph
    pub fn new() -> Self {
        Graph {
            edges: HashMap::new(),
        }
    }

    /// Adds a vertex to the graph
    pub fn add_vertex(&mut self, vertex: T) {
        self.edges.entry(vertex).or_default();
    }

    /// Adds a directed edge from source to destination
    pub fn add_edge(&mut self, source: T, destination: T) {
        self.edges
            .entry(source.clone())
            .or_default()
            .push(destination.clone());
        // Ensure the destination vertex exists in the graph
        self.edges.entry(destination).or_default();
    }

    /// Performs a depth-first search to find a path to the target vertex
    ///
    /// # Arguments
    /// * `start` - The vertex to start the search from
    /// * `target` - The vertex to search for
    ///
    /// # Returns
    /// * `Ok(Some(path))` - A vector representing the path from start to target
    /// * `Ok(None)` - Target vertex not found
    /// * `Err(Error)` - An error occurred during the search
    ///
    /// # Examples
    /// ```
    /// # use algos::cs::search::dfs::Graph;
    /// #
    /// let mut graph = Graph::new();
    /// graph.add_edge(1, 2);
    /// graph.add_edge(2, 3);
    /// graph.add_edge(1, 4);
    ///
    /// assert!(matches!(graph.search(&1, &3).unwrap(), Some(path) if path == vec![1, 2, 3]));
    /// assert!(matches!(graph.search(&1, &5).unwrap(), None));
    /// ```
    pub fn search(&self, start: &T, target: &T) -> Result<Option<Vec<T>>> {
        if !self.edges.contains_key(start) {
            return Err(Error::invalid_input("Start vertex not found in graph"));
        }

        let mut visited = HashSet::new();
        let mut path = Vec::new();

        if self.dfs_recursive(start, target, &mut visited, &mut path) {
            Ok(Some(path))
        } else {
            Ok(None)
        }
    }

    /// Recursive helper function for depth-first search
    fn dfs_recursive(
        &self,
        current: &T,
        target: &T,
        visited: &mut HashSet<T>,
        path: &mut Vec<T>,
    ) -> bool {
        visited.insert(current.clone());
        path.push(current.clone());

        if current == target {
            return true;
        }

        if let Some(neighbors) = self.edges.get(current) {
            for neighbor in neighbors {
                if !visited.contains(neighbor)
                    && self.dfs_recursive(neighbor, target, visited, path)
                {
                    return true;
                }
            }
        }

        path.pop();
        false
    }

    /// Performs an iterative depth-first search using a stack
    ///
    /// # Arguments
    /// * `start` - The vertex to start the search from
    /// * `target` - The vertex to search for
    ///
    /// # Returns
    /// * `Ok(Some(path))` - A vector representing the path from start to target
    /// * `Ok(None)` - Target vertex not found
    /// * `Err(Error)` - An error occurred during the search
    pub fn search_iterative(&self, start: &T, target: &T) -> Result<Option<Vec<T>>> {
        if !self.edges.contains_key(start) {
            return Err(Error::invalid_input("Start vertex not found in graph"));
        }

        let mut stack = vec![(start.clone(), vec![start.clone()])];
        let mut visited = HashSet::new();
        visited.insert(start.clone());

        while let Some((current, path)) = stack.pop() {
            if &current == target {
                return Ok(Some(path));
            }

            if let Some(neighbors) = self.edges.get(&current) {
                for neighbor in neighbors.iter().rev() {
                    // Reverse to maintain same order as recursive
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        let mut new_path = path.clone();
                        new_path.push(neighbor.clone());
                        stack.push((neighbor.clone(), new_path));
                    }
                }
            }
        }

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph() {
        let graph: Graph<i32> = Graph::new();
        assert!(matches!(graph.search(&1, &2), Err(Error::InvalidInput(_))));
    }

    #[test]
    fn test_single_vertex() {
        let mut graph = Graph::new();
        graph.add_vertex(1);
        assert!(matches!(graph.search(&1, &1).unwrap(), Some(path) if path == vec![1]));
        assert!(matches!(graph.search_iterative(&1, &1).unwrap(), Some(path) if path == vec![1]));
    }

    #[test]
    fn test_direct_edge() {
        let mut graph = Graph::new();
        graph.add_edge(1, 2);
        assert!(matches!(graph.search(&1, &2).unwrap(), Some(path) if path == vec![1, 2]));
        assert!(
            matches!(graph.search_iterative(&1, &2).unwrap(), Some(path) if path == vec![1, 2])
        );
    }

    #[test]
    fn test_path_not_found() {
        let mut graph = Graph::new();
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        assert!(matches!(graph.search(&1, &4).unwrap(), None));
        assert!(matches!(graph.search_iterative(&1, &4).unwrap(), None));
    }

    #[test]
    fn test_complex_path() {
        let mut graph = Graph::new();
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(2, 4);
        graph.add_edge(3, 5);
        graph.add_edge(4, 5);

        let path = graph.search(&1, &5).unwrap().unwrap();
        assert!(path.starts_with(&[1, 2]));
        assert_eq!(*path.last().unwrap(), 5);

        let path_iter = graph.search_iterative(&1, &5).unwrap().unwrap();
        assert!(path_iter.starts_with(&[1, 2]));
        assert_eq!(*path_iter.last().unwrap(), 5);
    }

    #[test]
    fn test_cyclic_graph() {
        let mut graph = Graph::new();
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 1);
        graph.add_edge(2, 4);

        assert!(matches!(graph.search(&1, &4).unwrap(), Some(path) if path == vec![1, 2, 4]));
        assert!(
            matches!(graph.search_iterative(&1, &4).unwrap(), Some(path) if path == vec![1, 2, 4])
        );
    }

    #[test]
    fn test_with_strings() {
        let mut graph = Graph::new();
        graph.add_edge("A", "B");
        graph.add_edge("B", "C");
        graph.add_edge("A", "D");

        assert!(
            matches!(graph.search(&"A", &"C").unwrap(), Some(path) if path == vec!["A", "B", "C"])
        );
        assert!(
            matches!(graph.search_iterative(&"A", &"C").unwrap(), Some(path) if path == vec!["A", "B", "C"])
        );
    }

    #[test]
    fn test_disconnected_components() {
        let mut graph = Graph::new();
        graph.add_edge(1, 2);
        graph.add_edge(3, 4);

        assert!(matches!(graph.search(&1, &4).unwrap(), None));
        assert!(matches!(graph.search_iterative(&1, &4).unwrap(), None));
    }

    #[test]
    fn test_invalid_start_vertex() {
        let mut graph = Graph::new();
        graph.add_edge(1, 2);

        assert!(matches!(graph.search(&3, &2), Err(Error::InvalidInput(_))));
        assert!(matches!(
            graph.search_iterative(&3, &2),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn test_multiple_paths() {
        let mut graph = Graph::new();
        graph.add_edge(1, 2);
        graph.add_edge(2, 4);
        graph.add_edge(1, 3);
        graph.add_edge(3, 4);

        // Both paths are valid, but DFS will find one of them
        let path = graph.search(&1, &4).unwrap().unwrap();
        assert!(
            path == vec![1, 2, 4] || path == vec![1, 3, 4],
            "Path should be either [1,2,4] or [1,3,4]"
        );

        let path_iter = graph.search_iterative(&1, &4).unwrap().unwrap();
        assert!(
            path_iter == vec![1, 2, 4] || path_iter == vec![1, 3, 4],
            "Path should be either [1,2,4] or [1,3,4]"
        );
    }
}
