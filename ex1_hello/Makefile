# Exercise 1: Hello OpenMP
# Makefile

# macOS OpenMP configuration
OMP_PREFIX = /opt/homebrew/opt/libomp
CC = clang
CFLAGS = -Wall -O2 -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include
LDFLAGS = -L$(OMP_PREFIX)/lib -lomp
TARGET = hello_omp

all: $(TARGET)

$(TARGET): hello_omp.c
	$(CC) $(CFLAGS) -o $(TARGET) hello_omp.c $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	@echo "Running with default threads:"
	./$(TARGET)
	@echo ""
	@echo "Running with 2 threads:"
	OMP_NUM_THREADS=2 ./$(TARGET)
	@echo ""
	@echo "Running with 4 threads:"
	OMP_NUM_THREADS=4 ./$(TARGET)
	@echo ""
	@echo "Running with 8 threads:"
	OMP_NUM_THREADS=8 ./$(TARGET)

.PHONY: all clean run
