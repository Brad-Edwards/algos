# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.8] - 2025-02-23

### Added

- Combinatorial algorithms:
  - Subset Generation
  - Zassenhaus Algorithm (group factorization)
  - Dancing Links (Algorithm X)
  - Gray Code Generation
  - Johnson-Trotter Algorithm
  - Backtracking for Permutations/Combinations
- Graph algorithms:
  - Edmonds' Blossom Algorithm (maximum matching)
  - Hamiltonian Cycle implementation

### Changed

- Moved Held-Karp algorithm from combinatorial to graph module

## [0.6.7] - 2025-02-17

### Added

- Randomized algorithms:
  - Karger's Min Cut
  - Randomized Delaunay Triangulation
  - Randomized Kruskal's MST
  - Randomized Prim's MST
  - Randomized BFS 2-SAT
  - Reservoir Sampling
  - Skip List
- Monte Carlo algorithms:
  - Monte Carlo Integration

### Changed

- Improved code quality with Clippy fixes
- Reorganized Monte Carlo algorithms into dedicated module
- Enhanced randomized algorithm implementations with better error handling

## [0.6.6] - 2025-02-17

### Changed

- Removed duplicate release job from CI workflow
- Simplified CI/CD pipeline by keeping release process only in release.yml

## [0.6.5] - 2025-02-17

### Added

- Integer Linear Programming algorithms:
  - Branch and Price
  - Branch and Cut
  - Branch and Bound
  - Dantzig-Wolfe Decomposition
  - Benders Decomposition
  - Lift and Project
  - Gomory Cuts
- Optimization algorithms:
  - Simplex Method
  - Interior Point Method
  - Genetic Algorithm
- Deep Learning algorithms
- Reinforcement Learning algorithms
- Approximation algorithms
- Security algorithms (DO NOT USE - for educational purposes only):
  - Twofish
  - DSA
  - Elliptic Curve
- Hashing algorithms

### Changed

- Improved code quality with Clippy fixes
- Updated dependencies:
  - thiserror from 1.0.69 to 2.0.11
  - cargo-tarpaulin from 0.27.3 to 0.31.5
  - softprops/action-gh-release from 1 to 2

### Fixed

- Various linting issues and code improvements
- Test failures in multiple algorithms
- Standardized constraint handling in column generation
- Improved numerical stability in optimization algorithms

## [0.6.4] - 2025-01-22

### Added

- Graph algorithms:
  - Bellman-Ford Algorithm
  - Dijkstra's Algorithm
  - Floyd-Warshall Algorithm
  - Johnson's Algorithm
  - Kruskal's Algorithm
  - Prim's Algorithm
  - Tarjan's Algorithm (SCC)
  - Topological Sort
  - Warshall's Algorithm

## [0.6.3] - 2025-01-21

### Fixed

- Release workflow now creates release and publishes crate

## [0.6.0] - 2025-01-21

### Added

- Algorithm categories specification
- String algorithms:
  - Aho-Corasick
  - Boyer-Moore
  - Knuth-Morris-Pratt (KMP)
  - Manacher's Algorithm
  - Rabin-Karp
  - Rolling Hash
  - Suffix Array
  - Suffix Automaton
  - Suffix Tree
  - Z-Algorithm
- Sorting algorithms:
  - BubbleSort
  - BucketSort
  - CountingSort
  - HeapSort
  - InsertionSort
  - MergeSort
  - QuickSort
  - RadixSort
  - SelectionSort
  - ShellSort
- Searching algorithms:
  - Binary Search
  - Breadth-First Search
  - Depth-First Search
  - Exponential Search
  - Fibonacci Search
  - Interpolation Search
  - Jump Search
  - Linear Search
  - Sublist Search
  - Ternary Search

### Changed

- Repository ownership transferred to [@Brad-Edwards](https://github.com/Brad-Edwards)
- Major restructuring of codebase organization
- Expanded project scope to include advanced algorithm categories
- Updated documentation and development roadmap
