"""Prompt templates for baselines: vanilla CoT, Chain-of-Draft, budget-forced."""
COT = "Let's think step by step.\n"
CHAIN_OF_DRAFT = ("Think step by step, but keep each thinking step to at most "
                  "five words. Return the final answer after '####'.\n")
BUDGET_FORCED = "Solve concisely in at most {budget} tokens.\n"
