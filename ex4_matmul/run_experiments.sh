 #!/bin/bash
# Run experiments for Exercise 4: Matrix Multiplication

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"
PLOTS_DIR="${SCRIPT_DIR}/../plots"

mkdir -p "$DATA_DIR" "$PLOTS_DIR"

echo "=== Matrix Multiplication OpenMP Experiments ==="
echo ""

# Build the program
make clean
make

echo "--- Running thread scaling experiments ---"
./matmul_omp > "$DATA_DIR/matmul_scaling.csv"
echo "Results saved to $DATA_DIR/matmul_scaling.csv"

echo ""
echo "--- Running scheduling experiments ---"
echo "schedule,chunk,time" > "$DATA_DIR/matmul_schedule.csv"

for chunk in 1 4 16 64 256; do
    for schedule in static dynamic guided; do
        export OMP_SCHEDULE="${schedule},${chunk}"
        OMP_NUM_THREADS=4 ./matmul_omp --schedule-test -c $chunk 2>/dev/null | \
            grep -E "^(STATIC|DYNAMIC|GUIDED):" | \
            awk -v sched=$schedule -v c=$chunk '{gsub(":", "", $1); print tolower($1) "," c "," $2}' \
            >> "$DATA_DIR/matmul_schedule.csv"
    done
done

echo "Scheduling results saved to $DATA_DIR/matmul_schedule.csv"
echo ""
echo "To generate plots, run: python3 ../scripts/plot_matmul.py"
