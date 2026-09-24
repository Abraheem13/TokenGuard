# Grader determinism

Recomputed every cached verdict from scratch with the symbolic grader. Disagreements, if any, are listed below; the analyses read the committed cache, so they do not change any reported number.

| quantity | value |
|---|---|
| stored verdicts | 6666 |
| recomputed | 6666 |
| disagreements | 6 |

- `\cot x` vs `\frac{1}{\sin x \cos x} - \frac{\sin x}{\cos x}`: stored True, recomputed False
- `\frac{\cos x}{\sin x}` vs `\frac{\cos^2 x}{\cos x \sin x}`: stored True, recomputed False
- `\cot x` vs `\frac{\cos x}{\sin x}`: stored True, recomputed False
- `\frac{\cos x}{\sin x}` vs `\cot x`: stored True, recomputed False
- `\cot x` vs `\frac{1 - \sin^2 x}{\cos x \sin x}`: stored True, recomputed False
- `\cot x` vs `\frac{1 - \sin^2 x}{\sin x \cos x}`: stored True, recomputed False
