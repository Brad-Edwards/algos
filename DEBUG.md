# Ford-Fulkerson Implementation Debug Log

## Current Status
- Two failing tests:
  1. `test_ford_fulkerson_simple`: Getting 17.0 instead of expected 19.0
  2. `test_ford_fulkerson_complex_graph`: Getting 15.0 instead of expected 20.0

## Test Case Analysis

### test_ford_fulkerson_simple
```rust
graph.add_edge('A', 'B', 10.0);  // Source -> B
graph.add_edge('A', 'C', 10.0);  // Source -> C
graph.add_edge('B', 'C', 2.0);   // B -> C
graph.add_edge('B', 'D', 8.0);   // B -> D
graph.add_edge('C', 'E', 9.0);   // C -> E
graph.add_edge('D', 'F', 10.0);  // D -> Sink
graph.add_edge('E', 'D', 4.0);   // E -> D
graph.add_edge('E', 'F', 10.0);  // E -> Sink
```

Expected paths:
1. A -> B -> D -> F (8 units)
2. A -> C -> E -> F (9 units)
3. A -> C -> E -> D -> F (2 units)
Total: 19 units

Actual paths found (from debug output):
1. A -> B -> D -> F (8 units)
2. A -> C -> E -> F (9 units)
Missing: A -> C -> E -> D -> F (2 units)

The algorithm is finding the first two paths but missing the third path that would give us the additional 2 units of flow. This path requires using the E -> D edge after we've already used E -> F.

### test_ford_fulkerson_complex_graph
```rust
graph.add_edge('S', 'A', 10.0);  // Source -> A
graph.add_edge('S', 'B', 5.0);   // Source -> B
graph.add_edge('A', 'C', 10.0);  // A -> C
graph.add_edge('A', 'D', 5.0);   // A -> D
graph.add_edge('B', 'C', 5.0);   // B -> C
graph.add_edge('B', 'D', 10.0);  // B -> D
graph.add_edge('C', 'T', 15.0);  // C -> Sink
graph.add_edge('D', 'T', 10.0);  // D -> Sink
```

Expected paths:
1. S -> A -> C -> T (10 units)
2. S -> B -> D -> T (5 units)
3. S -> A -> D -> T (5 units)
Total: 20 units

## Root Cause
The BFS is not finding paths that require "backtracking" through edges we've already used in a different direction. After we use an edge in one path, we're not properly considering it for use in the opposite direction in future paths.

## Next Steps
1. Modify BFS to consider both forward and backward edges when finding paths
2. Update residual graph handling to properly track both forward and backward capacities
3. Add more debug output to verify residual graph state after each path 