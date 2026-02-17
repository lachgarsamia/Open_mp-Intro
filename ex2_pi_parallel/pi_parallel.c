/*
 * Exercise 2: Parallelizing PI Calculation
 * 
 * This program calculates PI using numerical integration with OpenMP.
 * Uses #pragma omp parallel (not parallel for) with explicit work distribution.
 * Method: PI = integral from 0 to 1 of 4/(1+x^2) dx
 * 
 * Key concepts:
 * - Parallel construct
 * - Shared vs private variables
 * - Manual work distribution
 * - omp_get_wtime() for timing
 * 
 
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

static long num_steps = 100000000;  // 100 million steps for better timing
double step;

int main() {
    int i;
    double pi, sum = 0.0;
    double start_time, end_time;
    int num_threads = 1;
    const int cache_line_doubles = 8;  // 8 doubles = 64 bytes
    
    step = 1.0 / (double)num_steps;
    
    printf("=== PI Calculation using OpenMP Parallel Construct ===\n");
    printf("Number of steps: %ld\n\n", num_steps);
    
    // Sequential version for comparison
    start_time = omp_get_wtime();
    sum = 0.0;
    for (i = 0; i < num_steps; i++) {
        double x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }
    pi = step * sum;
    end_time = omp_get_wtime();
    
    printf("Sequential:\n");
    printf("  PI = %.15f\n", pi);
    printf("  Time = %.6f seconds\n\n", end_time - start_time);
    
    double seq_time = end_time - start_time;
    
    // Parallel version using omp parallel (NOT parallel for)
    // Each thread writes to a padded slot to avoid false sharing.
    start_time = omp_get_wtime();
    sum = 0.0;

    int max_threads = omp_get_max_threads();
    double *partial_sums = (double *)calloc((size_t)max_threads * cache_line_doubles, sizeof(double));
    if (!partial_sums) {
        fprintf(stderr, "Memory allocation failed for partial sums.\n");
        return EXIT_FAILURE;
    }

    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int nthrds = omp_get_num_threads();
        double local_sum = 0.0;  // Private variable for each thread
        
        // Only master thread saves num_threads
        if (id == 0) {
            num_threads = nthrds;
        }
        
        // Manual work distribution: thread i handles iterations i, i+nthrds, i+2*nthrds, ...
        for (int j = id; j < num_steps; j += nthrds) {
            double x = (j + 0.5) * step;
            local_sum += 4.0 / (1.0 + x * x);
        }

        partial_sums[id * cache_line_doubles] = local_sum;
    }

    for (int t = 0; t < num_threads; t++) {
        sum += partial_sums[t * cache_line_doubles];
    }
    free(partial_sums);
    
    pi = step * sum;
    end_time = omp_get_wtime();
    
    double par_time = end_time - start_time;
    double speedup = seq_time / par_time;
    double efficiency = speedup / num_threads * 100.0;
    
    printf("Parallel (with %d threads):\n", num_threads);
    printf("  PI = %.15f\n", pi);
    printf("  Time = %.6f seconds\n", par_time);
    printf("  Speedup = %.2fx\n", speedup);
    printf("  Efficiency = %.1f%%\n\n", efficiency);
    
    return 0;
}
