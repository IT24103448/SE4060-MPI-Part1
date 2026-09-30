#include <mpi.h>
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int number;

    if (rank == 0) {
        // Allocate enough buffer space for three outstanding messages.
        int packed_size;
        MPI_Pack_size(1, MPI_INT, MPI_COMM_WORLD, &packed_size);

        int buffer_size = 3 * (packed_size + MPI_BSEND_OVERHEAD);
        std::vector<char> buffer(buffer_size);
        MPI_Buffer_attach(buffer.data(), buffer_size);

        for (int i = 0; i < 3; ++i) {
            number = i * 10;

            MPI_Bsend(&number, 1, MPI_INT,
                      1, 0, MPI_COMM_WORLD);

            std::cout << "Process 0 sent " << number << "\n";
        }

        // Complete pending buffered sends before buffer storage is released.
        void* detached_buffer;
        int detached_size;
        MPI_Buffer_detach(&detached_buffer, &detached_size);

    } else if (rank == 1) {
        for (int i = 0; i < 3; ++i) {
            MPI_Recv(&number, 1, MPI_INT,
                     0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            std::cout << "Process 1 received " << number << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
