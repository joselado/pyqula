"""Entry point for the pyqula benchmark suite: runs a set of cases (each
comparing multiple methods for computing the same quantity) and writes +
compiles a LaTeX report. See documentation/benchmark_plan.md.

Usage (run from the repo root, same convention as `python -m pytest tests`):

    python -m benchmarks.run_all --quick             # default: small sizes
    python -m benchmarks.run_all --full               # larger sweep
    python -m benchmarks.run_all --case dos_methods    # restrict to one case
    python -m benchmarks.run_all --no-compile          # skip LaTeX compile

Not wired into pytest: benchmark timings are inherently machine-dependent
and not something a pass/fail test suite should assert on.
"""
import argparse
import os

from benchmarks import report
from benchmarks.cases import CASES
from benchmarks.harness import machine_info, save_records

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    sweep = parser.add_mutually_exclusive_group()
    sweep.add_argument("--quick", action="store_true",
                        help="run the small size sweep (default)")
    sweep.add_argument("--full", action="store_true",
                        help="run the larger size sweep instead of --quick")
    parser.add_argument("--case", action="append", dest="cases", metavar="NAME",
                         help="restrict to this case (repeatable); default: all cases")
    parser.add_argument("--no-compile", action="store_true",
                         help="write report.tex but skip LaTeX compilation")
    args = parser.parse_args()

    case_names = args.cases or list(CASES.keys())
    unknown = [c for c in case_names if c not in CASES]
    if unknown:
        parser.error(f"unknown case(s) {unknown}; available: {list(CASES.keys())}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    machine = machine_info()

    for name in case_names:
        case = CASES[name]
        sizes = case.SIZES_FULL if args.full else case.SIZES_QUICK
        print(f"Running {name} (sizes={sizes})...")
        records = case.run(sizes)
        save_records(RESULTS_DIR, name, records, machine)
        print(f"  -> {len(records)} records")

    tex_path = report.build_report(RESULTS_DIR, case_names)
    print(f"Wrote {tex_path}")
    if not args.no_compile:
        pdf_path = report.compile_report(tex_path)
        if pdf_path:
            print(f"Compiled {pdf_path}")


if __name__ == "__main__":
    main()
