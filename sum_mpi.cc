#include <mpi.h>
#include <iostream>
#include <iomanip>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const long long N = 10000000;

    // Divide the numbers evenly, including any remainder.
    long long base = N / size;
    long long remainder = N % size;
    long long count = base + (rank < remainder ? 1 : 0);

    long long first = rank * base
                    + (rank < remainder ? rank : remainder)
                    + 1;
    long long last = first + count - 1;

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Each process adds its assigned numbers.
    long long local_sum = 0;
    for (long long i = first; i <= last; ++i) {
        local_sum += i;
    }

    // Combine all local sums at rank 0.
    long long total_sum = 0;
    MPI_Reduce(&local_sum, &total_sum, 1,
               MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double local_elapsed = MPI_Wtime() - start_time;
    double elapsed = 0.0;

    MPI_Reduce(&local_elapsed, &elapsed, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long long expected = N * (N + 1) / 2;

        std::cout << "Processes: " << size << "\n"
                  << "Total sum: " << total_sum << "\n"
                  << "Expected:  " << expected << "\n"
                  << "Check: " << (total_sum == expected ? "PASS" : "FAIL")
                  << "\n"
                  << std::fixed << std::setprecision(6)
                  << "Time: " << elapsed << " seconds\n";
    }

    MPI_Finalize();
    return 0;
}
