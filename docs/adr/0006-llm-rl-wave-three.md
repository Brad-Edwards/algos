# ADR-0006: LLM RL Deferred to Wave 3

**Status:** Accepted

## Context

GRPO, PPO-RLHF, DPO, and REINFORCE++ are all RL algorithms applied to language model fine-tuning. They share infrastructure (KL divergence against a reference policy, reward normalization, reference policy abstraction) with each other and with general RL.

The question is whether to include these in Wave 2 or treat them as a separate wave.

## Decision

LLM RL algorithms (GRPO, PPO-RLHF, DPO, REINFORCE++) are Wave 3.

The shared infrastructure they depend on (KL divergence utility, reference policy abstraction, reward normalization) is Wave 2, because it is general-purpose and used by continuous control algorithms as well.

The boundary:
- **Wave 2:** KL divergence computation, reference policy trait, running reward normalization, group-level reward normalization
- **Wave 3:** GRPO, PPO-RLHF variant, DPO, REINFORCE++

## Consequences

- Wave 3 cannot begin until Wave 2 infrastructure (KL, reference policy, normalization) is complete.
- Wave 3 algorithms implement `Policy` from ADR-0003; the user provides an LLM that implements the trait.
- No CUDA or LLM-framework dependency enters the library; the trait boundary (ADR-0002) holds.
- DPO is included in Wave 3 despite being a supervised method because it uses the same `Policy` + reference policy + preference data interface as the other Wave 3 algorithms.
