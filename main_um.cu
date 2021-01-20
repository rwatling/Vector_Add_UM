// This program calculates the sum of two vectors using unifed memory
// By: Robbie Watling

# include "system_includes.h"

using namespace std;

// CUDA kernel for vector addition
__global__ void vectorAdd(int* a, int* b, int* c, int N) {
    // Calculate global thread thread ID
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

    // Boundary check
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    //Performance variables
    clock_t time_start;
    clock_t time_end;
    double elapsed;
    ofstream my_file;
    string run_file_name;


    run_file_name = "vector_add_um_performance.txt";
    my_file.open(run_file_name);

    for (int j = 0; j < 15; j++) {
        //Initialize clock (in ticks)
        time_start = clock();

        //Initialize array information
        int N = 1 << 14;
        size_t bytes = N * sizeof(int);

        // Declare unified memory pointers
        int* a, * b, * c;

        // Allocation memory for these pointers
        // Memory automatically managed
        cudaMallocManaged((void**) &a, bytes);
        cudaMallocManaged((void**) &b, bytes);
        cudaMallocManaged((void**) &c, bytes);

        // Initialize vectors
        srand(time(NULL));
        for (int i = 0; i < N; i++) {
            a[i] = rand() % 100;
            b[i] = rand() % 100;
        }

        // Threads per CTA (1024 threads per CTA)
        int BLOCK_SIZE = 1 << 4;

        // CTAs per Grid
        int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Call CUDA kernel
        vectorAdd <<<GRID_SIZE, BLOCK_SIZE >> > (a, b, c, N);

        // Wait for all previous operations before using values
        // We need this because we don't get the implicit synchronization of
        // cudaMemcpy like in the original example
        cudaDeviceSynchronize();

        // Verify the result on the CPU
        for (int i = 0; i < N; i++) {
            assert(c[i] == a[i] + b[i]);
        }

        // Free unified memory (same as memory allocated with cudaMalloc)
        cudaFree(a);
        cudaFree(b);
        cudaFree(c);

        cout << "COMPLETED SUCCESSFULLY!\n";

        time_end = clock();
        elapsed = (double)(time_end - time_start) / CLOCKS_PER_SEC;

        //Performance information
        my_file << "Number of array elements (n): " << N << endl;
        my_file << "Number of threads: " << BLOCK_SIZE / 1024 << endl;
        my_file << "Number of blocks: " << GRID_SIZE / BLOCK_SIZE << endl;
        my_file << "Total time: " << elapsed << " seconds" << endl;
        my_file << "---------------------------------------------" << endl;
    }

    my_file.close();

    return 0;
}