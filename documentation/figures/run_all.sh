#!/usr/bin/env bash
# Recompute every figure of the user guide, one script at a time.
#
#   bash documentation/figures/run_all.sh            # all figures
#   bash documentation/figures/run_all.sh kpm_*.py   # some of them
#
# Each script writes documentation/guide/img/<id>.png and <id>-dark.png.
# A failed script is reported at the end and does not stop the others.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
scripts=("$@")
[ ${#scripts[@]} -eq 0 ] && scripts=($(ls *.py | grep -v '^style\.py$'))
failed=()
for f in "${scripts[@]}"; do
    start=$(date +%s)
    if python3 "$f" > /dev/null; then
        echo "ok      $f ($(( $(date +%s) - start )) s)"
    else
        echo "FAILED  $f"; failed+=("$f")
    fi
done
[ ${#failed[@]} -eq 0 ] || { echo "failed: ${failed[*]}"; exit 1; }
