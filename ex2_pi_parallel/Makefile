# Exercise 2: PI Calculation with Parallel Construct
# Makefile

# macOS OpenMP configuration
OMP_PREFIX = /opt/homebrew/opt/libomp
CC = clang
CFLAGS = -Wall -O2 -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include
LDFLAGS = -L$(OMP_PREFIX)/lib -lomp -lm
TARGET = pi_parallel

all: $(TARGET)

$(TARGET): pi_parallel.c
	$(CC) $(CFLAGS) -o $(TARGET) pi_parallel.c $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	@echo "=== Running with different thread counts ==="
	@echo ""
	@echo "--- 1 thread ---"
	OMP_NUM_THREADS=1 ./$(TARGET)
	@echo "--- 2 threads ---"
	OMP_NUM_THREADS=2 ./$(TARGET)
	@echo "--- 4 threads ---"
	OMP_NUM_THREADS=4 ./$(TARGET)
	@echo "--- 8 threads ---"
	OMP_NUM_THREADS=8 ./$(TARGET)

.PHONY: all clean run
