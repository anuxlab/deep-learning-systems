#!/usr/bin/env python
"""Print a Markdown job summary for GitHub Actions: status, check count, timings, log tail."""
import json, sys
from pathlib import Path

nb = Path(sys.argv[1]); outcome = sys.argv[2] if len(sys.argv) > 2 else "unknown"
folder = nb.parent; log = folder / "run.log"; met = folder / "metrics.json"
icon = {"success": "✅", "failure": "❌"}.get(outcome, "ℹ️")
print(f"## {icon} `{nb.name}` — {outcome}\n")
text = log.read_text(errors="replace") if log.exists() else ""
print(f"- checks passed (`✅` lines in the log): **{text.count('✅')}**")
print(f"- figures saved: **{len(list((folder / 'figures').glob('*.png'))) if (folder / 'figures').exists() else 0}**")
if met.exists():
    m = json.loads(met.read_text()); t = m.get("timings_s", {})
    env = m.get("environment", {})
    print(f"- environment: Python {env.get('python')}, torch {env.get('torch')}, {env.get('cpu_count')} CPUs")
    if t:
        print("\n| timed section | seconds |\n|---|---|")
        for k, v in t.items(): print(f"| {k} | {v} |")
print("\n<details><summary>Last 60 lines of run.log</summary>\n\n```text")
print("\n".join(text.splitlines()[-60:]))
print("```\n</details>")
