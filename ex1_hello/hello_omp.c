

#include <stdio.h>
#include <omp.h>

int main() {
    int num_threads;
    
    // Start parallel region
    #pragma omp parallel
    {
        // Get thread information inside parallel region
        int thread_id = omp_get_thread_num();
        int total_threads = omp_get_num_threads();
        
        // Only master thread stores the total
        #pragma omp master
        {
            num_threads = total_threads;
        }
        
        // Each thread prints its greeting
        printf("Hello from the rank %d thread\n", thread_id);
    }
    
    // Print summary after parallel region
    printf("Parallel execution of hello_world with %d threads\n", num_threads);
    
    return 0;
}
