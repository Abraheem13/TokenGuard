# Grader reproducibility

Every cached verdict recomputed from scratch in one process, in cache order. Each disagreement is recomputed again in a fresh interpreter; `fresh` is that verdict. The analyses read the committed cache, so no reported number depends on this recomputation.

| quantity | value |
|---|---|
| stored verdicts | 8340 |
| recomputed | 8340 |
| disagreements | 8 |
| disagreements matching the stored verdict when recomputed in a fresh interpreter | 8 |

| prediction | reference | stored | sequential | fresh |
|---|---|---|---|---|
| `\cot x` | `\frac{1}{\sin x \cos x} - \frac{\sin x}{\cos x}` | True | False | True |
| `\frac{\cos x}{\sin x}` | `\frac{\cos^2 x}{\cos x \sin x}` | True | False | True |
| `\cot x` | `\frac{\cos x}{\sin x}` | True | False | True |
| `\frac{\cos x}{\sin x}` | `\cot x` | True | False | True |
| `\cot x` | `\frac{1 - \sin^2 x}{\cos x \sin x}` | True | False | True |
| `\cot x` | `\frac{1 - \sin^2 x}{\sin x \cos x}` | True | False | True |
| `\dfrac{11 + 9a}{20}` | `\frac{11+9a}{20}` | True | False | True |
| `\frac{\cos x}{\sin x}` | `\frac{1 - \sin^2 x}{\cos x \sin x}` | True | False | True |
