"""
Full-scope sweep matching the paper's own complete testbed:
    n in {30, 35, 40, 45, 50, 55}   (6 values)
    R in {5, 10, 30}                (3 values)
    alpha in {0.25, 0.50, 0.75}     (3 values)
    D in {25%, 50%, 75%, 100%}      (4 values)
    -> 6 x 3 x 3 x 4 = 216 configurations (exactly the paper's 216)

5 methods this time: standard, glover, ss, afg, direct
    -> 216 x 5 = 1080 total solves

Independent of full_sweep_n30.csv from the earlier (n=30-only, 4-method)
run -- writes to its own file, full_sweep_n30to55.csv, so nothing needs
to be deleted or merged. n=30 is re-solved here too (now including
"direct"), in its own self-contained dataset.

Same per-instance settings as before: TimeLimit=1200s, Threads=4.
Resumable: already-completed (n,R,alpha,D,method) rows are skipped on
re-run, so an interrupted sweep restarts exactly where it left off.

SCALE WARNING: 216 configs x 5 methods = 1080 solves, each capped at
1200s -> a 360-hour (15-day) worst case if every single one timed out.
This will NOT finish in one sitting. Expect to run it across many
sessions; resume handles that automatically. At n=55 in particular,
AFG's Stage-1 RLT model has ~2*55*54=5940 auxiliary variables (times 2
for w1/w2) -- large but solvable as an LP given a full/Academic Gurobi
license (the free size-limited license was the earlier blocker, not
solve time).

Usage:
    python3 full_sweep_n30to55.py
"""
import csv
import itertools
import os

import gurobipy as gp
from qmdkp_exact_methods import download_instance_file, load_instances, solve_instance

N_VALUES = [30, 35, 40, 45, 50, 55]
R_VALUES = [5, 10, 30]
ALPHA_VALUES = [25, 50, 75]
D_VALUES = [25, 50, 75, 100]
METHODS = ["standard", "glover", "ss", "afg", "direct"]
TIME_LIMIT = 1200.0
THREADS = 4
CSV_PATH = "full_sweep_n30to55.csv"


def already_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["n"], row["R"], row["alpha"], row["density"], row["method"]))
    return done


def main():
    write_header = not os.path.exists(CSV_PATH)
    done = already_done(CSV_PATH)

    configs = list(itertools.product(N_VALUES, R_VALUES, ALPHA_VALUES, D_VALUES))
    print(f"{len(configs)} parameter configurations x {len(METHODS)} methods "
          f"= {len(configs) * len(METHODS)} total solves\n")

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["n", "R", "alpha", "density", "method", "status",
                              "objective", "published_optimal", "gap_pct",
                              "runtime_s", "correct"])

        for n, R, alpha, D in configs:
            fname = f"qmdkp_n{n}_R{R}_d{D}_a{alpha}.json"
            try:
                path = download_instance_file(fname)
                instances = load_instances(path)
            except Exception as e:
                print(f"[SKIP] could not load {fname}: {e}")
                continue

            inst = instances["instance_1"]
            data = inst["instance_data"]
            published = inst["optimal_solution"]["obj_val"]

            for method in METHODS:
                key = (str(n), str(R), str(alpha), str(D), method)
                if key in done:
                    print(f"n={n:2d} R={R:2d} a={alpha:3d} D={D:3d}% {method:10s}: "
                          f"already done, skipping")
                    continue

                try:
                    r = solve_instance(data, method, time_limit=TIME_LIMIT, threads=THREADS)
                    gap = r["mip_gap"] * 100 if r["mip_gap"] is not None else None
                    status = "OPTIMAL" if r["solved"] else "TIME_LIMIT"
                    correct = (
                        r["obj_val"] is not None and abs(r["obj_val"] - published) < 1e-3
                    ) if status == "OPTIMAL" else "N/A (not proven optimal)"

                    writer.writerow([n, R, alpha, D, method, status, r["obj_val"],
                                      published, gap, r["time"], correct])
                    f.flush()
                    obj_str = f"{r['obj_val']:.2f}" if r["obj_val"] is not None else "None"
                    gap_str = f"{gap:.4f}%" if gap is not None else "N/A"
                    print(f"n={n:2d} R={R:2d} a={alpha:3d} D={D:3d}% {method:10s}: "
                          f"{status:10s} obj={obj_str} gap={gap_str} "
                          f"time={r['time']:.2f}s correct={correct}")
                except gp.GurobiError as e:
                    writer.writerow([n, R, alpha, D, method, "ERROR", "", published,
                                      "", "", str(e)])
                    f.flush()
                    print(f"n={n:2d} R={R:2d} a={alpha:3d} D={D:3d}% {method:10s}: ERROR -> {e}")

    print(f"\nDone (or paused). Results in {CSV_PATH}")


if __name__ == "__main__":
    main()
