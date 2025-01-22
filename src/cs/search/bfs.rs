use crate::cs::error::{Error, Result};
use std::{
    collections::{HashMap, HashSet, VecDeque},
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

    /// Performs a breadth-first search to find the shortest path to the target vertex
    ///
    /// # Arguments
    /// * `start` - The vertex to start the search from
    /// * `target` - The vertex to search for
    ///
    /// # Returns
    /// * `Ok(Some(path))` - A vector representing the shortest path from start to target
    /// * `Ok(None)` - Target vertex not found
    /// * `Err(Error)` - An error occurred during the search
    ///
    /// # Examples
    /// ```
    /// # use algos::cs::search::bfs::Graph;
    /// #
    /// let mut graph = Graph::new();
    /// graph.add_edge(1, 2);
    /// graph.add_edge(2, 3);
    /// graph.add_edge(1, 4);
    ///
    /// assert!(matches!(graph.search(&1, &3).unwrap(), Some(path) if path == vec![1, 2, 3]));
    /// assert!(matches!(graph.search(&1, &5).unwrap(), None));
    /// ```
    ///
    /// # Performance
    /// * Time: O(V + E) where V is the number of vertices and E is the number of edges
    /// * Space: O(V) for the queue and visited set
    pub fn search(&self, start: &T, target: &T) -> Result<Option<Vec<T>>> {
        if !self.edges.contains_key(start) {
            return Err(Error::invalid_input("Start vertex not found in graph"));
        }

        // If start is the target, return immediately
        if start == target {
            return Ok(Some(vec![start.clone()]));
        }

        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut parent = HashMap::new();

        queue.push_back(start.clone());
        visited.insert(start.clone());

        while let Some(current) = queue.pop_front() {
            if let Some(neighbors) = self.edges.get(&current) {
                for neighbor in neighbors {
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        parent.insert(neighbor.clone(), current.clone());
                        queue.push_back(neighbor.clone());

                        if neighbor == target {
                            // Reconstruct path from target to start
                            return Ok(Some(self.reconstruct_path(&parent, start, target)));
                        }
                    }
                }
            }
        }

        Ok(None)
    }

    /// Reconstructs the path from start to target using the parent map
    fn reconstruct_path(&self, parent: &HashMap<T, T>, start: &T, target: &T) -> Vec<T> {
        let mut path = Vec::new();
        let mut current = target.clone();
        path.push(current.clone());

        while current != *start {
            current = parent[&current].clone();
            path.push(current.clone());
        }

        path.reverse();
        path
    }

    /// Returns all vertices reachable from the start vertex in BFS order
    ///
    /// # Arguments
    /// * `start` - The vertex to start the traversal from
    ///
    /// # Returns
    /// * `Ok(vertices)` - A vector of vertices in BFS order
    /// * `Err(Error)` - An error occurred during the traversal
    pub fn traverse(&self, start: &T) -> Result<Vec<T>> {
        if !self.edges.contains_key(start) {
            return Err(Error::invalid_input("Start vertex not found in graph"));
        }

        let mut result = Vec::new();
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();

        queue.push_back(start.clone());
        visited.insert(start.clone());
        result.push(start.clone());

        while let Some(current) = queue.pop_front() {
            if let Some(neighbors) = self.edges.get(&current) {
                for neighbor in neighbors {
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        queue.push_back(neighbor.clone());
                        result.push(neighbor.clone());
                    }
                }
            }
        }

        Ok(result)
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
    }

    #[test]
    fn test_direct_edge() {
        let mut graph = Graph::new();
        graph.add_edge(1, 2);
        assert!(matches!(graph.search(&1, &2).unwrap(), Some(path) if path == vec![1, 2]));
    }

    #[test]
    fn test_path_not_found() {
        let mut graph = Graph::new();
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        assert!(matches!(graph.search(&1, &4).unwrap(), None));
    }

    #[test]
    fn test_shortest_path() {
        let mut graph = Graph::new();
        // Path 1: 1 -> 2 -> 3 -> 4 (length 4)
        // Path 2: 1 -> 5 -> 4 (length 3)
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 4);
        graph.add_edge(1, 5);
        graph.add_edge(5, 4);

        assert!(matches!(graph.search(&1, &4).unwrap(), Some(path) if path == vec![1, 5, 4]));
    }

    #[test]
    fn test_cyclic_graph() {
        let mut graph = Graph::new();
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 1);
        graph.add_edge(2, 4);

        assert!(matches!(graph.search(&1, &4).unwrap(), Some(path) if path == vec![1, 2, 4]));
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
    }

    #[test]
    fn test_disconnected_components() {
        let mut graph = Graph::new();
        graph.add_edge(1, 2);
        graph.add_edge(3, 4);

        assert!(matches!(graph.search(&1, &4).unwrap(), None));
    }

    #[test]
    fn test_invalid_start_vertex() {
        let mut graph = Graph::new();
        graph.add_edge(1, 2);

        assert!(matches!(graph.search(&3, &2), Err(Error::InvalidInput(_))));
    }

    #[test]
    fn test_bfs_traversal_order() {
        let mut graph = Graph::new();
        graph.add_edge(1, 2);
        graph.add_edge(1, 3);
        graph.add_edge(2, 4);
        graph.add_edge(2, 5);
        graph.add_edge(3, 6);

        let traversal = graph.traverse(&1).unwrap();
        // BFS should visit vertices level by level
        assert_eq!(traversal[0], 1); // Level 0
        assert!(traversal[1..3].contains(&2) && traversal[1..3].contains(&3)); // Level 1
        assert!(traversal[3..].iter().all(|&x| x >= 4)); // Level 2
    }

    #[test]
    fn test_multiple_paths_same_length() {
        let mut graph = Graph::new();
        // Two paths of length 3: 1->2->4 and 1->3->4
        graph.add_edge(1, 2);
        graph.add_edge(2, 4);
        graph.add_edge(1, 3);
        graph.add_edge(3, 4);

        let path = graph.search(&1, &4).unwrap().unwrap();
        assert_eq!(path.len(), 3); // Should find a path of length 3
        assert!(
            (path == vec![1, 2, 4]) || (path == vec![1, 3, 4]),
            "Path should be either [1,2,4] or [1,3,4]"
        );
    }
}
