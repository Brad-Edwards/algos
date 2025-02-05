# Algos

[![Crates.io](https://img.shields.io/crates/v/algos.svg)](https://crates.io/crates/algos)
[![Documentation](https://docs.rs/algos/badge.svg)](https://docs.rs/algos)
[![CI](https://github.com/Brad-Edwards/algos/actions/workflows/ci.yml/badge.svg)](https://github.com/Brad-Edwards/algos/actions/workflows/ci.yml)
[![GitHub license](https://img.shields.io/github/license/Brad-Edwards/algos.svg)](https://github.com/Brad-Edwards/algos/blob/master/LICENSE)

## 🚧 Work in Progress 🚧

This crate is undergoing significant development. It aims to be a comprehensive collection of algorithms implemented in Rust, serving both as a learning resource and a practical library.

## Recent Changes

**Jan 16, 2025 - ADVISORY**: This crate has changed hands and will be maintained by [@Brad-Edwards](https://github.com/Brad-Edwards). The repository has been moved to [a new location](https://github.com/Brad-Edwards/algos).

## Overview

A Rust library implementing a wide range of algorithms, from fundamental computer science concepts to advanced machine learning techniques.

## Implementation Status

### ✅ Currently Implemented

#### Sorting Algorithms

✅ QuickSort  
✅ MergeSort  
✅ HeapSort  
✅ BubbleSort  
✅ InsertionSort  
✅ SelectionSort  
✅ ShellSort  
✅ CountingSort  
✅ RadixSort  
✅ BucketSort  

#### Searching Algorithms

✅ Linear Search  
✅ Binary Search  
✅ Ternary Search  
✅ Interpolation Search  
✅ Jump Search  
✅ Exponential Search  
✅ Fibonacci Search  
✅ Sublist Search  
✅ Depth-First Search  
✅ Breadth-First Search  

#### String Algorithms

✅ Knuth-Morris-Pratt (KMP)  
✅ Rabin-Karp  
✅ Boyer-Moore  
✅ Z-Algorithm  
✅ Aho-Corasick  
✅ Suffix Array Construction  
✅ Suffix Automaton  
✅ Suffix Tree  
✅ Rolling Hash  
✅ Manacher's Algorithm  

#### Graph Algorithms (Part 1)

✅ Dijkstra's Algorithm  
✅ Bellman-Ford Algorithm  
✅ Floyd-Warshall Algorithm  
✅ Prim's Algorithm  
✅ Kruskal's Algorithm  
✅ Tarjan's Algorithm (SCC)  
✅ Kosaraju's Algorithm  
✅ Johnson's Algorithm  
✅ Warshall's Algorithm  
✅ Topological Sort  

#### Graph Algorithms (Part 2)

✅ Edmond–Karp (max flow)  
✅ Dinic’s Algorithm (max flow)  
✅ Ford–Fulkerson (max flow)  
✅ Hungarian Algorithm (assignment)  
✅ Hopcroft–Karp (bipartite matching)  
✅ Bron–Kerbosch (maximal clique)  
✅ Johnson’s Cycle Detection  
✅ Floyd’s Cycle Detection (Tortoise and Hare)  
✅ Euler Tour / Euler Circuit Algorithm  
✅ Hierholzer’s Algorithm (Euler paths/circuits)

#### Dynamic Programming

✅ Kadane's Algorithm  
✅ Matrix Chain Multiplication  
✅ Edit Distance  
✅ Coin Change  
✅ Longest Common Subsequence  
✅ Longest Increasing Subsequence  
✅ Weighted Interval Scheduling  
✅ Viterbi Algorithm  
✅ Bellman Equation-based DP  
✅ Knuth Optimization  

#### Hashing

✅ Perfect Hashing  
✅ Universal Hashing  
✅ Cuckoo Hashing  
✅ Separate Chaining  
✅ Open Addressing (linear/quadratic probing, double hashing)  
✅ Polynomial Rolling Hash  
✅ FNV (Fowler–Noll–Vo) Hash  
✅ CRC32  
✅ Jenkins Hash  
✅ MurmurHash  

#### Classic Machine Learning

✅ k-Means Clustering
✅ k-Nearest Neighbors (k-NN)
✅ Linear Regression (OLS)
✅ Logistic Regression
✅ Decision Tree Learning (ID3, C4.5)
✅ Random Forest
✅ Support Vector Machine (SVM)
✅ Naive Bayes
✅ Gradient Boosting (GBM family)
✅ XGBoost

#### ***DO NOT USE*** Cryptography and Security ***DO NOT USE***

✅ RSA  
✅ Diffie–Hellman Key Exchange  
✅ ElGamal Encryption  
✅ AES (Rijndael)  
✅ Blowfish  
✅ Twofish  
✅ SHA-256  
✅ MD5 (legacy)  
✅ Elliptic Curve Cryptography (ECC)  
✅ DSA (Digital Signature Algorithm)  

#### Deep Learning & Neural Network Training

✅ Backpropagation  
✅ Stochastic Gradient Descent (SGD)  
✅ Adam Optimizer  
✅ RMSProp  
✅ AdaGrad  
✅ RProp (resilient propagation)  
✅ Dropout (regularization)  
✅ Batch Normalization  
✅ Convolution (core of CNNs)  
✅ BPTT (Backprop Through Time, RNNs/LSTMs)  

#### Reinforcement Learning

✅ Q-Learning  
✅ SARSA  
✅ Double Q-Learning  
✅ Deep Q-Network (DQN)  
✅ Monte Carlo Exploring Starts  
✅ Policy Gradients (REINFORCE)  
✅ Actor–Critic Methods  
✅ Proximal Policy Optimization (PPO)  
✅ Trust Region Policy Optimization (TRPO)  
✅ AlphaZero-Style MCTS + RL  

#### Approximation Algorithms

✅ Greedy Set Cover  
✅ 2-Approximation for Vertex Cover  
✅ PTAS for Knapsack  
✅ Christofides’ Algorithm (TSP)  
✅ Johnson’s Algorithm for MAX-SAT  
✅ FPTAS for Subset Sum  
✅ Local-Ratio Theorem  
✅ Primal–Dual Approximation (for covering problems)  
✅ LP Rounding (generic approach)  
✅ Goemans–Williamson (Max-Cut)  

#### Linear & Nonlinear Optimization

✅ Gradient Descent  
✅ Newton’s Method  
✅ Conjugate Gradient  
✅ BFGS  
✅ L-BFGS  
✅ Simplex Method (linear programming)  
✅ Interior Point Method (LP/NLP)  
✅ Nelder–Mead  
✅ Genetic Algorithm  
✅ Simulated Annealing  

### 🚧 Planned Implementation

See our [Algorithm Categories](notes/specifications/ALGORITHM-CATEGORIES.md) document for the full list of planned implementations across all categories, including:

- Reinforcement Learning
- Approximation Algorithms
- Linear & Nonlinear Optimization
- And many more...

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.
