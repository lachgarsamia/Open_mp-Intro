#!/bin/bash
# Run experiments for Exercise 5: Jacobi Method

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"
PLOTS_DIR="${SCRIPT_DIR}/../plots"

mkdir -p "$DATA_DIR" "$PLOTS_DIR"

echo "=== Jacobi Method OpenMP Experiments ==="
echo ""

# Build the program with larger system size
make clean
make N_SIZE=1000

echo "--- Running scaling experiments ---"
./jacobi_omp --scaling > "$DATA_DIR/jacobi_scaling.csv"
echo "Results saved to $DATA_DIR/jacobi_scaling.csv"

echo ""
cat "$DATA_DIR/jacobi_scaling.csv"

echo ""
echo "To generate plots, run: python3 ../scripts/plot_jacobi.py"
