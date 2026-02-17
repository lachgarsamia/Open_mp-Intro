# Exercise 3: PI Calculation with Loop Construct
# Makefile

# macOS OpenMP configuration
OMP_PREFIX = /opt/homebrew/opt/libomp
CC = clang
CFLAGS = -Wall -O2 -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include
LDFLAGS = -L$(OMP_PREFIX)/lib -lomp -lm
TARGET = pi_loop

all: $(TARGET)

$(TARGET): pi_loop.c
	$(CC) $(CFLAGS) -o $(TARGET) pi_loop.c $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	@echo "=== Running with different thread counts ==="
	@for threads in 1 2 4 8 16; do \
		echo ""; \
		echo "--- $$threads thread(s) ---"; \
		OMP_NUM_THREADS=$$threads ./$(TARGET); \
	done

.PHONY: all clean run
