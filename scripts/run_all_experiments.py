"""Run every experiment in order and drop all artifacts into results/:
the GA fitness curve (+ best genome, needed by the other scripts), the
round-robin win-rate matrix (+ heatmap), the belief-convergence plot, the
Bayesian-model ablation, and the optional exploitability estimates.

Each step is a separate script/process so their argparse CLIs don't
collide; this file just sequences them with sane defaults.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent

STEPS = [
    ("genetic algorithm", ["run_ga.py", "--population", "40", "--generations", "30", "--hands-per-opponent", "300"]),
    ("round-robin win-rate matrix", ["run_baseline_eval.py", "--hands", "2000"]),
    ("belief convergence", ["plot_belief_convergence.py"]),
    ("ablation (search with vs without Bayes)", ["ablation.py"]),
    ("exploitability (optional)", ["run_exploitability.py"]),
]


def main():
    for label, cmd in STEPS:
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
        t0 = time.time()
        subprocess.run([sys.executable, str(SCRIPTS_DIR / cmd[0]), *cmd[1:]], check=True)
        print(f"[{label} finished in {time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
