/*
 * Exercise 4: Matrix Multiplication with OpenMP
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

// Default matrix dimensions
#ifndef M_SIZE
#define M_SIZE 1024
#endif

#ifndef N_SIZE
#define N_SIZE 1024
#endif

void init_matrices(double *a, double *b, double *c, int m, int n) {
    // Initialize matrix A (m x n)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            a[i * n + j] = (i + 1) + (j + 1);
        }
    }
    
    // Initialize matrix B (n x m)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            b[i * m + j] = (i + 1) - (j + 1);
        }
    }
    
    // Initialize matrix C (m x m) to zeros
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            c[i * m + j] = 0;
        }
    }
}

// Sequential matrix multiplication
double matmul_sequential(double *a, double *b, double *c, int m, int n) {
    double start = omp_get_wtime();
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            c[i * m + j] = 0;
            for (int k = 0; k < n; k++) {
                c[i * m + j] += a[i * n + k] * b[k * m + j];
            }
        }
    }
    
    return omp_get_wtime() - start;
}

// Parallel matrix multiplication - basic parallel for
double matmul_parallel_basic(double *a, double *b, double *c, int m, int n) {
    double start = omp_get_wtime();
    
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            c[i * m + j] = 0;
            for (int k = 0; k < n; k++) {
                c[i * m + j] += a[i * n + k] * b[k * m + j];
            }
        }
    }
    
    return omp_get_wtime() - start;
}

// Parallel matrix multiplication with collapse(2)
double matmul_parallel_collapse(double *a, double *b, double *c, int m, int n) {
    double start = omp_get_wtime();
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += a[i * n + k] * b[k * m + j];
            }
            c[i * m + j] = sum;
        }
    }
    
    return omp_get_wtime() - start;
}

// Parallel with schedule(static, chunk)
double matmul_parallel_static(double *a, double *b, double *c, int m, int n, int chunk) {
    double start = omp_get_wtime();
    
    #pragma omp parallel for schedule(static, chunk) collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += a[i * n + k] * b[k * m + j];
            }
            c[i * m + j] = sum;
        }
    }
    
    return omp_get_wtime() - start;
}

// Parallel with schedule(dynamic, chunk)
double matmul_parallel_dynamic(double *a, double *b, double *c, int m, int n, int chunk) {
    double start = omp_get_wtime();
    
    #pragma omp parallel for schedule(dynamic, chunk) collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += a[i * n + k] * b[k * m + j];
            }
            c[i * m + j] = sum;
        }
    }
    
    return omp_get_wtime() - start;
}

// Parallel with schedule(guided, chunk)
double matmul_parallel_guided(double *a, double *b, double *c, int m, int n, int chunk) {
    double start = omp_get_wtime();
    
    #pragma omp parallel for schedule(guided, chunk) collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += a[i * n + k] * b[k * m + j];
            }
            c[i * m + j] = sum;
        }
    }
    
    return omp_get_wtime() - start;
}

int main(int argc, char *argv[]) {
    int m = M_SIZE;
    int n = N_SIZE;
    int num_threads = omp_get_max_threads();
    int run_mode = 0;  // 0: full analysis, 1: single run, 2: schedule test
    int chunk_size = 16;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            m = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            num_threads = atoi(argv[++i]);
            omp_set_num_threads(num_threads);
        } else if (strcmp(argv[i], "--single") == 0) {
            run_mode = 1;
        } else if (strcmp(argv[i], "--schedule-test") == 0) {
            run_mode = 2;
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            chunk_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -m SIZE       Matrix M dimension (default: %d)\n", M_SIZE);
            printf("  -n SIZE       Matrix N dimension (default: %d)\n", N_SIZE);
            printf("  -t THREADS    Number of threads\n");
            printf("  -c CHUNK      Chunk size for scheduling\n");
            printf("  --single      Run single comparison test\n");
            printf("  --schedule-test  Run scheduling comparison\n");
            return 0;
        }
    }
    
    // Allocate memory
    double *a = (double *)malloc(m * n * sizeof(double));
    double *b = (double *)malloc(n * m * sizeof(double));
    double *c = (double *)malloc(m * m * sizeof(double));
    
    if (!a || !b || !c) {
        fprintf(stderr, "Memory allocation failed!\n");
        return EXIT_FAILURE;
    }
    
    printf("=== Matrix Multiplication with OpenMP ===\n");
    printf("Matrix A: %d x %d\n", m, n);
    printf("Matrix B: %d x %d\n", n, m);
    printf("Matrix C: %d x %d\n", m, m);
    printf("Threads: %d\n\n", num_threads);
    
    if (run_mode == 1) {
        // Single run mode - compare sequential vs parallel
        init_matrices(a, b, c, m, n);
        double seq_time = matmul_sequential(a, b, c, m, n);
        printf("Sequential time: %.4f s\n", seq_time);
        
        init_matrices(a, b, c, m, n);
        double par_time = matmul_parallel_collapse(a, b, c, m, n);
        printf("Parallel time (collapse): %.4f s\n", par_time);
        printf("Speedup: %.2fx\n", seq_time / par_time);
        printf("Efficiency: %.1f%%\n", (seq_time / par_time) / num_threads * 100);
        
    } else if (run_mode == 2) {
        // Schedule test mode
        printf("=== Scheduling Comparison (chunk=%d) ===\n", chunk_size);
        
        init_matrices(a, b, c, m, n);
        double static_time = matmul_parallel_static(a, b, c, m, n, chunk_size);
        printf("STATIC:  %.4f s\n", static_time);
        
        init_matrices(a, b, c, m, n);
        double dynamic_time = matmul_parallel_dynamic(a, b, c, m, n, chunk_size);
        printf("DYNAMIC: %.4f s\n", dynamic_time);
        
        init_matrices(a, b, c, m, n);
        double guided_time = matmul_parallel_guided(a, b, c, m, n, chunk_size);
        printf("GUIDED:  %.4f s\n", guided_time);
        
    } else {
        // Full analysis mode - output CSV format for plotting
        printf("threads,sequential,parallel_basic,parallel_collapse,speedup_basic,speedup_collapse,efficiency_basic,efficiency_collapse\n");
        
        int thread_counts[] = {1, 2, 4, 8, 16};
        int num_tests = 5;
        
        // Get sequential baseline
        init_matrices(a, b, c, m, n);
        double seq_time = matmul_sequential(a, b, c, m, n);
        
        for (int t = 0; t < num_tests; t++) {
            int threads = thread_counts[t];
            if (threads > omp_get_max_threads()) continue;
            
            omp_set_num_threads(threads);
            
            init_matrices(a, b, c, m, n);
            double basic_time = matmul_parallel_basic(a, b, c, m, n);
            
            init_matrices(a, b, c, m, n);
            double collapse_time = matmul_parallel_collapse(a, b, c, m, n);
            
            double speedup_basic = seq_time / basic_time;
            double speedup_collapse = seq_time / collapse_time;
            double eff_basic = speedup_basic / threads * 100;
            double eff_collapse = speedup_collapse / threads * 100;
            
            printf("%d,%.4f,%.4f,%.4f,%.2f,%.2f,%.1f,%.1f\n",
                   threads, seq_time, basic_time, collapse_time,
                   speedup_basic, speedup_collapse, eff_basic, eff_collapse);
        }
    }
    
    free(a);
    free(b);
    free(c);
    
    return EXIT_SUCCESS;
}
