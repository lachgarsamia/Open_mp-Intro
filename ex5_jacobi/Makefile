# Exercise 5: Jacobi Method with OpenMP
# Makefile

# macOS OpenMP configuration
OMP_PREFIX = /opt/homebrew/opt/libomp
CC = clang
CFLAGS = -Wall -O2 -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include
LDFLAGS = -L$(OMP_PREFIX)/lib -lomp -lm
TARGET = jacobi_omp

# Default system size
N_SIZE ?= 1000
D_SIZE ?= 80

all: $(TARGET)

$(TARGET): jacobi_omp.c
	$(CC) $(CFLAGS) -DVAL_N=$(N_SIZE) -DVAL_D=$(D_SIZE) -o $(TARGET) jacobi_omp.c $(LDFLAGS)

small: jacobi_omp.c
	$(CC) $(CFLAGS) -DVAL_N=120 -DVAL_D=80 -o $(TARGET)_small jacobi_omp.c $(LDFLAGS)

large: jacobi_omp.c
	$(CC) $(CFLAGS) -DVAL_N=2000 -DVAL_D=80 -o $(TARGET)_large jacobi_omp.c $(LDFLAGS)

clean:
	rm -f $(TARGET) $(TARGET)_small $(TARGET)_large *.csv

run: $(TARGET)
	./$(TARGET)

scaling: $(TARGET)
	./$(TARGET) --scaling

test: $(TARGET)
	@echo "=== Testing with different thread counts ==="
	@for threads in 1 2 4 8; do \
		echo ""; \
		echo "--- $$threads thread(s) ---"; \
		OMP_NUM_THREADS=$$threads ./$(TARGET); \
	done

.PHONY: all clean run scaling test small large
