# TP3 - OpenMP Introduction

**Author:** Implementation based on TP3 by Prof.Imad Kissami  
**Date:** February 2026

## Overview

This lab introduces OpenMP (Open Multi-Processing), a parallel programming API for shared-memory multiprocessor systems. The exercises cover fundamental OpenMP concepts including:

- Thread creation and management
- Parallel regions
- Work sharing constructs
- Synchronization
- Performance analysis

## Prerequisites

- GCC compiler with OpenMP support (`gcc` with `-fopenmp` flag)
- Python 3 with matplotlib for plotting
- Basic understanding of C programming

## Exercises

### Exercise 1: Hello OpenMP
Introduction to OpenMP parallel regions, displaying thread ranks and counts.

### Exercise 2: Pi Calculation (Parallel Construct)
Parallelizing Pi calculation using `#pragma omp parallel` with explicit work distribution (no `parallel for`), padded per-thread partial sums, and serial merge (no `critical`/`atomic` accumulation).

### Exercise 3: Pi with Loops
Parallelizing Pi calculation using `#pragma omp parallel for` with minimal code changes.

### Exercise 4: Matrix Multiplication
- OpenMP parallelization with `collapse` directive
- Performance analysis with different thread counts (1, 2, 4, 8, 16)
- Scheduling strategies comparison (STATIC, DYNAMIC, GUIDED)
- Speedup and efficiency plots

### Exercise 5: Jacobi Iterative Method
- Parallel implementation of Jacobi method for solving linear systems
- Single persistent `#pragma omp parallel` region around the iterative loop, with `omp for` worksharing and `single` convergence/copy steps
- Performance analysis with different thread counts
- Speedup and efficiency analysis

## Directory Structure

```
openmp-intro/
├── README.md
├── ex1_hello/
│   ├── hello_omp.c
│   └── Makefile
├── ex2_pi_parallel/
│   ├── pi_parallel.c
│   └── Makefile
├── ex3_pi_loop/
│   ├── pi_loop.c
│   └── Makefile
├── ex4_matmul/
│   ├── matmul_omp.c
│   ├── Makefile
│   └── run_experiments.sh
├── ex5_jacobi/
│   ├── jacobi_omp.c
│   ├── Makefile
│   └── run_experiments.sh
├── scripts/
│   ├── plot_matmul.py
│   └── plot_jacobi.py
└── data/
    └── (generated results)
```

## Building and Running

### Quick Start

```bash
# Build all exercises
./build_all.sh

# Run individual exercises
cd ex1_hello && ./hello_omp
cd ex2_pi_parallel && ./pi_parallel
cd ex3_pi_loop && ./pi_loop
cd ex4_matmul && ./run_experiments.sh
cd ex5_jacobi && ./run_experiments.sh
```

### Setting Thread Count

```bash
export OMP_NUM_THREADS=4
./executable_name
```

## Key OpenMP Concepts

### Parallel Region
```c
#pragma omp parallel
{
    // Code executed by all threads
}
```

### Work Sharing (for loop)
```c
#pragma omp parallel for
for (int i = 0; i < n; i++) {
    // Iterations distributed among threads
}
```

### Reduction
```c
#pragma omp parallel for reduction(+:sum)
for (int i = 0; i < n; i++) {
    sum += array[i];
}
```

### Critical Section
```c
#pragma omp critical
{
    // Only one thread at a time
}
```

## Performance Metrics

- **Speedup:** S(p) = T(1) / T(p)
- **Efficiency:** E(p) = S(p) / p

Where T(1) is sequential execution time and T(p) is parallel execution time with p threads.

All exercise timings are measured with `omp_get_wtime()`.
