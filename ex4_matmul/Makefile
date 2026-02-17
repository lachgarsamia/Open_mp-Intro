# Exercise 4: Matrix Multiplication with OpenMP
# Makefile

# macOS OpenMP configuration
OMP_PREFIX = /opt/homebrew/opt/libomp
CC = clang
CFLAGS = -Wall -O2 -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include
LDFLAGS = -L$(OMP_PREFIX)/lib -lomp -lm
TARGET = matmul_omp

# Default matrix size
M_SIZE ?= 1024
N_SIZE ?= 1024

all: $(TARGET)

$(TARGET): matmul_omp.c
	$(CC) $(CFLAGS) -DM_SIZE=$(M_SIZE) -DN_SIZE=$(N_SIZE) -o $(TARGET) matmul_omp.c $(LDFLAGS)

small: matmul_omp.c
	$(CC) $(CFLAGS) -DM_SIZE=512 -DN_SIZE=512 -o $(TARGET)_small matmul_omp.c $(LDFLAGS)

clean:
	rm -f $(TARGET) $(TARGET)_small *.csv

run: $(TARGET)
	@./$(TARGET)

test: $(TARGET)
	@echo "=== Single Thread vs Multi-Thread Test ==="
	OMP_NUM_THREADS=4 ./$(TARGET) --single

schedule: $(TARGET)
	@echo "=== Scheduling Strategy Comparison ==="
	@echo ""
	@echo "--- Chunk size: 1 ---"
	OMP_NUM_THREADS=4 ./$(TARGET) --schedule-test -c 1
	@echo ""
	@echo "--- Chunk size: 16 ---"
	OMP_NUM_THREADS=4 ./$(TARGET) --schedule-test -c 16
	@echo ""
	@echo "--- Chunk size: 64 ---"
	OMP_NUM_THREADS=4 ./$(TARGET) --schedule-test -c 64
	@echo ""
	@echo "--- Chunk size: 256 ---"
	OMP_NUM_THREADS=4 ./$(TARGET) --schedule-test -c 256

.PHONY: all clean run test schedule small
