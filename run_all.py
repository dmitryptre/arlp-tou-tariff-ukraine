"""Run the whole computation of the article from its inputs, in order.

    python run_all.py                 # every step; stops at the first failure
    python run_all.py --from-step 3   # restart from a step (the outputs of the earlier steps must be in place)

The data that are not in the repository must be in place first (see README.md, "Data"). Every step writes into
outputs/ARLP_Clustering/.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
OUT = ROOT / "outputs" / "ARLP_Clustering"

STEPS = [
    ("hourly DAM prices in UAH (inputs/hourly_dam_prices_uah.csv)", ["build_dam_prices_uah.py"]),
    ("monthly BASE_M prices in UAH (inputs/base_m_prices_uah.csv)", ["build_base_m_prices_uah.py"]),
    ("notebook: clustering, classes, tariff, demand response, bills, PSO compensation",
     ["-m", "nbconvert", "--to", "notebook", "--execute", "--ExecutePreprocessor.timeout=3600",
      "--ExecutePreprocessor.kernel_name=python3", "--output-dir", str(OUT / "notebook_run"),
      "--output", "ARLP_clustering_analysis_autonomous.ipynb", "ARLP_clustering_analysis_autonomous.ipynb"]),
    ("within-class distribution of passive bill changes", ["run_eulp_bill_distribution.py"]),
    ("shift and combined effects within the classes", ["run_eulp_bill_response.py"]),
    ("sensitivity of the tariff multipliers", ["run_robustness_eulp.py"]),
    ("tables and figures of the sensitivity study", ["make_robustness_outputs.py", str(OUT / "robustness_eulp")]),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-step", type=int, default=1)
    a = ap.parse_args(argv)
    env = dict(os.environ, PYTHONUTF8="1")
    t_all = time.time()
    for i, (what, args) in enumerate(STEPS, start=1):
        if i < a.from_step:
            continue
        print(f"step {i}/{len(STEPS)}: {what}", flush=True)
        t = time.time()
        p = subprocess.run([sys.executable, "-X", "utf8", *args], cwd=CODE, env=env, check=False)
        print(f"step {i}: exit {p.returncode} in {time.time() - t:.0f} s", flush=True)
        if p.returncode != 0:
            print(f"stopped at step {i}")
            return 1
    print(f"done in {time.time() - t_all:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
