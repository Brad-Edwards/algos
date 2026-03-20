# Repository Vision: Strategic Analysis

## What This Repo Is, Actually

**247 algorithms. 879 passing tests. 50k lines of Rust. Built entirely with AI assistance via Cursor.**

The .cursorrules file reveals the development approach: an AI agent with a scratchpad, working through algorithm categories systematically, fixing failures, documenting lessons, moving on. The result is broad but honest — when implementations hit known limits (PPM, Branch & Price), the lessons were recorded rather than hidden.

All tests pass. The code is clippy-clean, consistently documented with complexity analysis, and generically typed. It's a real codebase, not a toy.

The honest question is: **useful for whom, and for what?**

---

## What Exists in the Rust Ecosystem Already

Before deciding what this should be, look at what already exists and is production-quality:

| Domain | What's already out there |
|--------|--------------------------|
| Graph algorithms | `petgraph` — dominant, production-quality, widely used |
| Classic ML (KNN, SVM, trees, regression) | `linfa` — scikit-learn equivalent, multi-contributor, maintained |
| Deep learning | `candle` (HuggingFace), `burn` — real frameworks |
| Sorting/searching | `std::collections`, iterators — good enough for most uses |
| Compression | `flate2`, `brotli`, `zstd` — battle-tested bindings |
| ECC/hashing | Specialized crates (`reed-solomon-erasure`, `crc`, etc.) |
| String algorithms | Scattered but covered (`aho-corasick` crate is dominant) |

**The unmet needs in the ecosystem:**

| Domain | Gap | This repo's coverage |
|--------|-----|----------------------|
| Reinforcement learning | Almost nothing dominant in Rust — `gym-rs`, `rurel`, scattered | Q-Learning, SARSA, DQN, PPO, TRPO, Actor-Critic, AlphaZero MCTS — 10+ algorithms |
| Approximation algorithms | Nearly absent in Rust | Christofides, Goemans-Williamson, PTAS, LP rounding — 10 algorithms |
| Integer linear programming | Very sparse (no pure-Rust dominant solver) | Branch & Bound, Benders, Column Generation, Gomory cuts — 10 methods |
| Algorithm education in Rust | Fragmented (TheAlgorithms/Rust exists but is shallower) | Comprehensive with complexity docs |

---

## The Core Tension

This repo tries to be everything:
- A **reference implementation** (use this as a starting point)
- An **educational resource** (learn how algorithms work)
- A **practical library** (pull this in as a dependency)

These goals partially conflict. Production libraries ruthlessly narrow scope. Educational resources prioritize explanation over generality. Reference libraries require correctness guarantees.

Right now the repo leans toward **reference implementation** — the implementations are real, the tests are real, the docs explain the algorithms. But it doesn't fully commit to that role.

---

## Three Viable Paths Forward

### Path A: "CLRS in Rust" — The Algorithm Education Library

Lean fully into education. Restructure around teaching rather than interface. Add:
- Explanatory module docs that walk through the algorithm (not just complexity tables)
- Step-by-step trace capabilities (the algorithm running verbosely)
- Visualization-ready data structures (algorithm state that can be serialized/rendered)
- Companion write-ups or links to canonical explanations

**Differentiation from TheAlgorithms/Rust:** Depth over breadth. Fewer algorithms but genuinely understood, with the complexity and trade-offs explained. The current repo already has better docs than TheAlgorithms — lean into that.

**Problem:** This requires fundamentally changing the API design. The current API is functional-library style, not pedagogical-tool style. Major work.

### Path B: "Fill the RL Gap" — The Rust RL Library

Pivot heavily toward reinforcement learning, which has the clearest ecosystem gap. This means:
- Define proper traits: `Environment`, `Agent`, `Policy`, `ReplayBuffer`
- Make the RL implementations actually composable (right now each is standalone)
- Add integration with real environments (at minimum, a GridWorld; ideally OpenAI Gym-compatible)
- Proper neural network integration for DQN/PPO/TRPO (right now they use toy approximators)

**Differentiation:** First serious RL library in Rust with real algorithmic depth. The current implementations (PPO, TRPO, AlphaZero MCTS) are non-trivial — no other Rust crate has these.

**Problem:** Requires significant API redesign of the RL section and proper tensor/array support. The current implementations are mostly self-contained demos.

### Path C: "Correct by Construction" — The Verified Algorithm Survey

Double down on correctness rather than breadth. The scarce resource in algorithm libraries isn't "implementations" — it's **verified-correct** implementations with known complexity and edge case behavior.

This means:
- Property-based testing (using `proptest`) for every algorithm
- Proven correctness properties documented and tested (sort stability, output invariants, convergence guarantees)
- Explicit marking of approximation ratios and when they hold
- Remove or prominently mark incomplete implementations
- Differential testing against known-good implementations where possible

**Differentiation:** Not just "here's QuickSort" but "here's QuickSort with proofs of average-case behavior verified by tests." Academics and serious practitioners care about this. Nothing in the ecosystem does this systematically.

**Problem:** Hard to do rigorously for advanced algorithms (PPO convergence guarantees are a research topic).

---

## What the Current Implementations Map To

Looking at what's here honestly:

**Genuinely useful as-is:**
- Sorting, searching, basic string algorithms — correct, generic, well-tested
- Hashing variants — useful reference for hash table design
- Classic DP problems — correct implementations of canonical problems
- Classic ML — correct but `linfa` is better for production; useful for learning

**Potentially the most valuable in the ecosystem:**
- RL algorithms — the deepest coverage of RL in Rust anywhere
- Approximation algorithms — essentially absent elsewhere in Rust
- ILP methods — sparse in Rust (though some have known limitations)

**Redundant with better-maintained alternatives:**
- Graph algorithms — `petgraph` wins for production
- Compression — use real crates
- Cryptography — the DO NOT USE warning is correct; `ring`/`rustls` win

**Educational value is real but incomplete:**
- Complexity docs are good
- No explanatory prose about *why* algorithms work
- No cross-linking between related algorithms (e.g., how Dijkstra relates to Prim's)

---

## A Proposed Purpose Statement

> **`algos` is a reference implementation library for algorithms in Rust, with emphasis on correctness, documentation, and coverage of advanced algorithms poorly served by the existing Rust ecosystem — particularly reinforcement learning, approximation algorithms, and combinatorial optimization.**
>
> It is not a production cryptography library, not a graph processing framework, and not a machine learning framework. For those, use `ring`, `petgraph`, and `linfa` respectively.
>
> The primary users are:
> 1. Rust developers learning how algorithms work with runnable, tested code
> 2. Researchers or engineers who need a starting-point implementation in Rust for RL or optimization
> 3. Contributors to the Rust ecosystem building domain-specific libraries

---

## What's Needed Now

Regardless of which path is chosen, several things would immediately improve the repo's usefulness:

1. **Acknowledge the AI-generated nature clearly.** Not as a disclaimer, but as context. This is an AI-assisted survey, not a team of domain experts. That shapes how users should engage with it.

2. **Fix the known-incomplete implementations.** The .cursorrules documents Branch & Price as having unfixed test failures (the `test_simple_ilp` case). PPM is described as a "simplified placeholder." These should be fixed or explicitly marked as incomplete in the code.

3. **Restructure the README around the ecosystem gaps.** Instead of listing 247 checkboxes, lead with: "If you need X in Rust and nothing exists, here's what we have." The RL and approximation sections deserve front-page positioning.

4. **Define the RL interfaces properly.** Even without full redesign, a minimal `Environment` trait would make the RL implementations composable and signal that this section has architectural intent.

5. **Remove the boilerplate badge-checklist structure** from the README and replace it with something that helps users actually decide whether this library is what they need.

---

## Summary Verdict

This repo has genuine value and covers real ground. The quality is better than it might appear — 879 tests pass, the code is idiomatic Rust with real error handling and generics. The scope is extraordinary for an AI-assisted project.

The risk is diffusion: being too broad to be trusted for any specific use case, competing with better-maintained libraries in well-served areas, while underselling the genuinely unique coverage in RL and approximation algorithms.

The highest-value path is probably **Path B + Path C combined**: commit to being the serious RL library in Rust (the gap is real) while establishing a correctness standard (property-based tests, explicit invariants) that makes the implementations trustworthy. Everything else — sorting, classic graph algorithms, basic ML — is fine to keep as educational context but shouldn't be the lead story.

The audience is real: Rust developers doing research, building agents, implementing optimization — they have nowhere else to go.
