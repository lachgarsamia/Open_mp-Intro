/*
 * Exercise 5: Jacobi Iterative Method with OpenMP
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#ifndef VAL_N
#define VAL_N 120
#endif

#ifndef VAL_D
#define VAL_D 80
#endif

void random_number(double* array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = (double)rand() / (double)(RAND_MAX - 1);
    }
}

// Sequential Jacobi solver
double jacobi_sequential(double *a, double *b, double *x, double *x_new, int n, int *iterations) {
    double start = omp_get_wtime();
    int iteration = 0;
    double norme;
    
    // Initialize x
    for (int i = 0; i < n; i++) {
        x[i] = 1.0;
    }
    
    while (1) {
        iteration++;
        
        // Compute new values
        for (int i = 0; i < n; i++) {
            x_new[i] = 0;
            for (int j = 0; j < i; j++) {
                x_new[i] += a[j * n + i] * x[j];
            }
            for (int j = i + 1; j < n; j++) {
                x_new[i] += a[j * n + i] * x[j];
            }
            x_new[i] = (b[i] - x_new[i]) / a[i * n + i];
        }
        
        // Compute norm
        double absmax = 0;
        for (int i = 0; i < n; i++) {
            double curr = fabs(x[i] - x_new[i]);
            if (curr > absmax)
                absmax = curr;
        }
        norme = absmax / n;
        
        if ((norme <= DBL_EPSILON) || (iteration >= n)) break;
        
        memcpy(x, x_new, n * sizeof(double));
    }
    
    *iterations = iteration;
    return omp_get_wtime() - start;
}

// Parallel Jacobi solver
double jacobi_parallel(double *a, double *b, double *x, double *x_new, int n, int *iterations) {
    double start = omp_get_wtime();
    int iteration = 0;
    double norme = 0.0;
    double absmax = 0.0;
    int stop = 0;

    // Single persistent team: avoid entering/leaving parallel regions in each iteration.
    #pragma omp parallel shared(a, b, x, x_new, n, iteration, norme, absmax, stop)
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++) {
            x[i] = 1.0;
        }

        while (1) {
            #pragma omp single
            {
                iteration++;
                absmax = 0.0;
            }

            #pragma omp for schedule(static)
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                for (int j = 0; j < i; j++) {
                    sum += a[j * n + i] * x[j];
                }
                for (int j = i + 1; j < n; j++) {
                    sum += a[j * n + i] * x[j];
                }
                x_new[i] = (b[i] - sum) / a[i * n + i];
            }

            #pragma omp for reduction(max:absmax) schedule(static)
            for (int i = 0; i < n; i++) {
                double curr = fabs(x[i] - x_new[i]);
                if (curr > absmax) {
                    absmax = curr;
                }
            }

            #pragma omp single
            {
                norme = absmax / n;
                if ((norme <= DBL_EPSILON) || (iteration >= n)) {
                    stop = 1;
                } else {
                    memcpy(x, x_new, (size_t)n * sizeof(double));
                }
            }

            if (stop) {
                break;
            }
        }
    }
    
    *iterations = iteration;
    return omp_get_wtime() - start;
}

int main(int argc, char *argv[]) {
    int n = VAL_N;
    int diag = VAL_D;
    int run_mode = 0;  // 0: single test, 1: scaling analysis
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            diag = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--scaling") == 0) {
            run_mode = 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -n SIZE       System size (default: %d)\n", VAL_N);
            printf("  -d DIAG       Diagonal dominance (default: %d)\n", VAL_D);
            printf("  --scaling     Run scaling analysis\n");
            return 0;
        }
    }
    
    // Allocate memory
    double *a = (double*)malloc(n * n * sizeof(double));
    double *x = (double*)malloc(n * sizeof(double));
    double *x_new = (double*)malloc(n * sizeof(double));
    double *b = (double*)malloc(n * sizeof(double));
    
    if (!a || !x || !x_new || !b) {
        fprintf(stderr, "Memory allocation failed!\n");
        return EXIT_FAILURE;
    }
    
    // Initialize
    srand(421);
    random_number(a, n * n);
    random_number(b, n);
    
    // Add diagonal dominance
    for (int i = 0; i < n; i++) {
        a[i * n + i] += diag;
    }
    
    if (run_mode == 0) {
        // Single test mode
        int num_threads = omp_get_max_threads();
        int seq_iter, par_iter;
        
        printf("=== Jacobi Method with OpenMP ===\n");
        printf("System size: %d\n", n);
        printf("Diagonal dominance: %d\n", diag);
        printf("Threads: %d\n\n", num_threads);
        
        // Sequential
        double seq_time = jacobi_sequential(a, b, x, x_new, n, &seq_iter);
        printf("Sequential:\n");
        printf("  Iterations: %d\n", seq_iter);
        printf("  Time: %.6f sec\n\n", seq_time);
        
        // Parallel
        double par_time = jacobi_parallel(a, b, x, x_new, n, &par_iter);
        double speedup = seq_time / par_time;
        double efficiency = speedup / num_threads * 100;
        
        printf("Parallel:\n");
        printf("  Iterations: %d\n", par_iter);
        printf("  Time: %.6f sec\n", par_time);
        printf("  Speedup: %.2fx\n", speedup);
        printf("  Efficiency: %.1f%%\n", efficiency);
        
    } else {
        // Scaling analysis mode
        printf("threads,sequential_time,parallel_time,iterations,speedup,efficiency\n");
        
        int thread_counts[] = {1, 2, 4, 8, 16};
        int num_tests = 5;
        
        // Get sequential baseline
        int seq_iter;
        double seq_time = jacobi_sequential(a, b, x, x_new, n, &seq_iter);
        
        for (int t = 0; t < num_tests; t++) {
            int threads = thread_counts[t];
            if (threads > omp_get_max_threads()) continue;
            
            omp_set_num_threads(threads);
            
            // Re-initialize
            srand(421);
            random_number(a, n * n);
            random_number(b, n);
            for (int i = 0; i < n; i++) {
                a[i * n + i] += diag;
            }
            
            int par_iter;
            double par_time = jacobi_parallel(a, b, x, x_new, n, &par_iter);
            
            double speedup = seq_time / par_time;
            double efficiency = speedup / threads * 100;
            
            printf("%d,%.6f,%.6f,%d,%.2f,%.1f\n",
                   threads, seq_time, par_time, par_iter, speedup, efficiency);
        }
    }
    
    free(a);
    free(x);
    free(x_new);
    free(b);
    
    return EXIT_SUCCESS;
}
