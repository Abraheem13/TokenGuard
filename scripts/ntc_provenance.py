#!/usr/bin/env python
"""Generation settings of every committed probe file, read from the file itself.

Each probe file stores the arguments of the run that produced it. This script
lists, per file, the model, benchmark, items, thinking budget, checkpoint
schedule, sampling seed and temperature, and whether the answer-format
instruction was appended to the questions (read from the stored questions), and
writes the scripts/ntc_w1_thinking.py command that regenerates the file.

Runs recorded before the harness had a --temperature option used its default
sampling, temperature 0.6. GPQA-Diamond files whose correct option is always A
present the options in source order, as DEER's own data file does; their
command carries --source-order.

Writes experiments/ntc/PROVENANCE.md.

    python scripts/ntc_provenance.py
"""
from __future__ import annotations

import json
from pathlib import Path

NTC = Path(__file__).resolve().parents[1] / "experiments" / "ntc"
INSTRUCTION_END = "within \\boxed{}."


def command(name: str, a: dict, instructed: bool, source_order: bool) -> str:
    parts = ["python scripts/ntc_w1_thinking.py",
             f"--model {a['model']}", f"--benchmark {a['benchmark']}",
             f"--limit {a['limit']}", f"--max-think {a['max_think']}"]
    if a.get("probe_every", 256) != 256 or a.get("max_probes", 10) != 10:
        parts += [f"--probe-every {a['probe_every']}", f"--max-probes {a['max_probes']}"]
    parts += [f"--max-model-len {a['max_model_len']}", f"--seed {a['seed']}"]
    temp = a.get("temperature")
    if temp is not None and temp != 0.6:
        parts.append(f"--temperature {temp}")
    if not instructed:
        parts.append("--no-instruction")
    if source_order:
        parts.append("--source-order")
    parts.append(f"--out experiments/ntc/{name}")
    return " ".join(parts)


def main() -> int:
    md = ["# Provenance of the probe files", "",
          "Settings read from the arguments stored in each probe file. `T` is the sampling "
          "temperature of the thinking pass (probes are always greedy); `instruction` "
          "records whether the answer-format instruction was appended to each question; "
          "`options` gives the GPQA-Diamond option order. "
          "All files were generated with vLLM on single NVIDIA L40S GPUs.", "",
          "| probe file | model | benchmark | items | budget | checkpoints | seed | T "
          "| instruction | options |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    cmds = []
    for f in sorted(NTC.glob("*.json")):
        d = json.loads(f.read_text())
        if "traces" not in d:
            continue
        a = d["args"]
        instructed = d["traces"][0]["question"].rstrip().endswith(INSTRUCTION_END)
        source_order = (a["benchmark"] == "gpqa_diamond"
                        and all(t["gold"] == "A" for t in d["traces"]))
        temp = a.get("temperature")
        t_str = "0.6" if temp is None else f"{temp:g}"
        options = ("n/a" if a["benchmark"] != "gpqa_diamond"
                   else "source order" if source_order else "shuffled")
        md.append(f"| `{f.name}` | {a['model'].split('/')[-1]} | {a['benchmark']} "
                  f"| {len(d['traces'])} | {a['max_think']} "
                  f"| every {a['probe_every']} tok, at most {a['max_probes']} "
                  f"| {a['seed']} | {t_str} | {'yes' if instructed else 'no'} "
                  f"| {options} |")
        cmds.append(command(f.name, a, instructed, source_order))
    md += ["", "## Regeneration commands", "",
           "Each command reproduces one file's generation settings; it needs a GPU "
           "and vLLM (see README).", "", "```bash"] + cmds + ["```"]
    out = NTC / "PROVENANCE.md"
    out.write_text("\n".join(md) + "\n")
    print(f"{len(cmds)} probe files\ntable: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
