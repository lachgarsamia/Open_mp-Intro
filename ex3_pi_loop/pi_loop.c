/*
 * Exercise 3: PI Calculation with Loop Construct
 */

#include <stdio.h>
#include <omp.h>

static long num_steps = 100000000;  // 100 million steps
double step;

int main() {
    int i;
    double x, pi, sum = 0.0;
    double start_time, end_time;
    
    step = 1.0 / (double)num_steps;
    
    printf("=== PI Calculation using OpenMP Parallel For ===\n");
    printf("Number of steps: %ld\n", num_steps);
    printf("Number of threads: %d\n\n", omp_get_max_threads());
    
    start_time = omp_get_wtime();
    
    #pragma omp parallel for private(x) reduction(+:sum)
    for (i = 0; i < num_steps; i++) {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }
    
    pi = step * sum;
    
    end_time = omp_get_wtime();
    
    printf("PI = %.15f\n", pi);
    printf("Time = %.6f seconds\n", end_time - start_time);
    
    return 0;
}
