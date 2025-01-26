use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct BronKerbosch {
    graph: HashMap<usize, Vec<usize>>,
}

impl BronKerbosch {
    pub fn new() -> Self {
        BronKerbosch {
            graph: HashMap::new(),
        }
    }

    pub fn add_edge(&mut self, u: usize, v: usize) {
        self.graph.entry(u).or_default().push(v);
        self.graph.entry(v).or_default().push(u);
    }

    pub fn find_cliques(&self) -> Vec<Vec<usize>> {
        let mut cliques = Vec::new();
        let vertices: HashSet<usize> = self.graph.keys().cloned().collect();
        self.bron_kerbosch(HashSet::new(), vertices, HashSet::new(), &mut cliques);
        cliques
    }

    fn bron_kerbosch(
        &self,
        r: HashSet<usize>,
        mut p: HashSet<usize>,
        mut x: HashSet<usize>,
        cliques: &mut Vec<Vec<usize>>,
    ) {
        if p.is_empty() && x.is_empty() {
            if !r.is_empty() {
                let mut clique: Vec<usize> = r.iter().cloned().collect();
                clique.sort_unstable();
                cliques.push(clique);
            }
            return;
        }

        // Choose pivot from P ∪ X to minimize branching
        let pivot = {
            let mut best_pivot = None;
            let mut max_neighbors = 0;

            for v in p.iter().chain(x.iter()) {
                if let Some(neighbors) = self.graph.get(v) {
                    let count = neighbors.iter().filter(|&n| p.contains(n)).count();
                    if count > max_neighbors {
                        max_neighbors = count;
                        best_pivot = Some(*v);
                    }
                }
            }
            best_pivot.unwrap_or_else(|| *p.iter().next().unwrap())
        };

        // Get neighbors of pivot
        let pivot_neighbors = if let Some(neighbors) = self.graph.get(&pivot) {
            neighbors.iter().cloned().collect::<HashSet<_>>()
        } else {
            HashSet::new()
        };

        let candidates: Vec<usize> = p
            .iter()
            .filter(|&v| !pivot_neighbors.contains(v))
            .cloned()
            .collect();

        for v in candidates {
            let v_neighbors: HashSet<usize> = self
                .graph
                .get(&v)
                .map(|neighbors| neighbors.iter().cloned().collect())
                .unwrap_or_default();

            let mut new_r = r.clone();
            new_r.insert(v);

            let new_p = p
                .iter()
                .filter(|&n| v_neighbors.contains(n))
                .cloned()
                .collect();

            let new_x = x
                .iter()
                .filter(|&n| v_neighbors.contains(n))
                .cloned()
                .collect();

            self.bron_kerbosch(new_r, new_p, new_x, cliques);

            p.remove(&v);
            x.insert(v);
        }
    }
}

impl Default for BronKerbosch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bron_kerbosch_find_cliques() {
        let mut bron_kerbosch = BronKerbosch::new();
        bron_kerbosch.add_edge(0, 1);
        bron_kerbosch.add_edge(0, 2);
        bron_kerbosch.add_edge(1, 2);
        let mut cliques = bron_kerbosch.find_cliques();
        cliques.sort_unstable();
        assert_eq!(cliques.len(), 1);
        assert!(cliques.contains(&vec![0, 1, 2]));
    }

    #[test]
    fn test_bron_kerbosch_find_cliques_2() {
        let mut bron_kerbosch = BronKerbosch::new();
        bron_kerbosch.add_edge(0, 1);
        bron_kerbosch.add_edge(1, 2);
        bron_kerbosch.add_edge(2, 3);
        let mut cliques = bron_kerbosch.find_cliques();
        cliques.sort_unstable();
        assert_eq!(cliques.len(), 3);
        assert!(cliques.contains(&vec![0, 1]));
        assert!(cliques.contains(&vec![1, 2]));
        assert!(cliques.contains(&vec![2, 3]));
    }

    #[test]
    fn test_bron_kerbosch_find_cliques_3() {
        let mut bron_kerbosch = BronKerbosch::new();
        bron_kerbosch.add_edge(0, 1);
        bron_kerbosch.add_edge(0, 2);
        bron_kerbosch.add_edge(1, 3);
        bron_kerbosch.add_edge(2, 3);
        let mut cliques = bron_kerbosch.find_cliques();
        cliques.sort_unstable();
        assert_eq!(cliques.len(), 4);
        assert!(cliques.contains(&vec![0, 1]));
        assert!(cliques.contains(&vec![0, 2]));
        assert!(cliques.contains(&vec![1, 3]));
        assert!(cliques.contains(&vec![2, 3]));
    }

    #[test]
    fn test_bron_kerbosch_find_cliques_empty() {
        let bron_kerbosch = BronKerbosch::new();
        let cliques = bron_kerbosch.find_cliques();
        assert_eq!(cliques.len(), 0);
    }
}
