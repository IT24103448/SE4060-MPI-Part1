#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const long long N = 10000000;
    const int TAG = 100;

    long long local_trials =
        N / size + (rank < N % size ? 1 : 0);

    std::mt19937 generator(12345u + static_cast<unsigned>(rank));
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    // Each worker sends one count.
    std::vector<char> buffer;
    if (rank != 0) {
        int packed_size;
        MPI_Pack_size(1, MPI_LONG_LONG_INT,
                      MPI_COMM_WORLD, &packed_size);

        int buffer_size = packed_size + MPI_BSEND_OVERHEAD;
        buffer.resize(buffer_size);
        MPI_Buffer_attach(buffer.data(), buffer_size);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    long long local_inside = 0;
    for (long long i = 0; i < local_trials; ++i) {
        double x = distribution(generator);
        double y = distribution(generator);

        if (x * x + y * y <= 1.0) {
            ++local_inside;
        }
    }

    long long total_inside = local_inside;

    if (rank == 0) {
        for (int i = 1; i < size; ++i) {
            long long received;

            MPI_Recv(&received, 1, MPI_LONG_LONG_INT,
                     MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

            total_inside += received;
        }
    } else {
        MPI_Bsend(&local_inside, 1, MPI_LONG_LONG_INT,
                  0, TAG, MPI_COMM_WORLD);

        // Include completion of buffered sends in the measured time.
        void* detached_buffer;
        int detached_size;
        MPI_Buffer_detach(&detached_buffer, &detached_size);
    }

    double local_elapsed = MPI_Wtime() - start_time;
    double elapsed = 0.0;

    MPI_Reduce(&local_elapsed, &elapsed, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double pi = 4.0 * static_cast<double>(total_inside) / N;

        std::cout << "Processes: " << size << "\n"
                  << "Total trials: " << N << "\n"
                  << "Points inside: " << total_inside << "\n"
                  << std::fixed << std::setprecision(8)
                  << "Estimated pi: " << pi << "\n"
                  << "Time: " << elapsed << " seconds\n";
    }

    MPI_Finalize();
    return 0;
}
