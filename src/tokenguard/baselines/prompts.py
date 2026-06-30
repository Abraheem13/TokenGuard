"""Prompt templates for baselines: vanilla CoT, Chain-of-Draft, budget-forced."""

COT = ("Solve this step by step. Be concise and do not second-guess yourself. "
       "When you reach the answer, stop and write it on a new line as: "
       "#### <answer>\n")

CHAIN_OF_DRAFT = ("Think step by step, but keep each thinking step to at most "
                  "five words. When you reach the answer, write it as: "
                  "#### <answer>\n")

BUDGET_FORCED = "Solve concisely in at most {budget} tokens. End with: #### <answer>\n"
