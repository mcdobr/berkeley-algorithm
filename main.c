#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

static const int MASTER_RANK = 0;
static const int POLL_TAG = 1;
static const int TIME_GATHER_TAG = 2;
static const int TIME_SYNC_TAG = 3;
static const int DELTA_TAG = 4;

static const char *poll_message = "TIME_POLL";


MPI_Status slave_wait_for_master_poll();

void slave_send_current_time(double start_time);

double average(int number_of_elements, const double *array);

int main(int argc, char *argv[]) {
    double start_time = MPI_Wtime();

    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == MASTER_RANK) {
        for (int destination_rank = 0; destination_rank < world_size; ++destination_rank) {
            MPI_Send(poll_message, 1 + strlen(poll_message), MPI_CHAR, destination_rank, POLL_TAG, MPI_COMM_WORLD);
        }


        double *times_received = malloc(world_rank * sizeof(double));
        double *round_trip_times = malloc(world_size * sizeof(double));

        MPI_Status status;

        /* TODO: the round trip times will be monotonically increasing (if msg #1 block but #2 would have been
         * received beforehand) but the times would be #1 < #2, which may lead to incorrect results (depends on the
         * desired precision)
         */
        for (int source_rank = 0; source_rank < world_size; ++source_rank) {
            MPI_Recv(&times_received[source_rank], 1, MPI_DOUBLE, source_rank, TIME_GATHER_TAG, MPI_COMM_WORLD, &status);
            round_trip_times[source_rank] = MPI_Wtime() - times_received[source_rank];
        }

        double correct_time = average(world_size, round_trip_times);

        for (int dest_rank = 0; dest_rank < world_size; ++dest_rank) {
            double delta = correct_time - times_received[dest_rank] + round_trip_times[dest_rank];
            MPI_Send(&delta, 1, MPI_DOUBLE, dest_rank, TIME_SYNC_TAG, MPI_COMM_WORLD);
        }

        free(times_received);
        times_received = NULL;

        free(round_trip_times);
        round_trip_times = NULL;


    } else {
        MPI_Status status = slave_wait_for_master_poll();
        slave_send_current_time(start_time);


        double delta;
        MPI_Recv(&delta, 1, MPI_DOUBLE, MASTER_RANK, DELTA_TAG, MPI_COMM_WORLD, &status);
        printf("Process [%d] received a delta of %.2lf\n", world_rank, delta);
    }


    MPI_Finalize();
    return 0;
}

double average(int number_of_elements, const double *array) {
    double result = 0.0;
    for (int i = 0; i < number_of_elements; ++i) {
        result += 1.0 / number_of_elements * array[i];
    }
    return result;
}

MPI_Status slave_wait_for_master_poll() {
    const int MAX_RECEIVED_SIZE = 256;
    MPI_Status status;

    char *buffer = malloc(MAX_RECEIVED_SIZE);
    MPI_Recv(buffer, MAX_RECEIVED_SIZE, MPI_CHAR, MASTER_RANK, POLL_TAG, MPI_COMM_WORLD, &status);
    free(buffer);
    buffer = NULL;
    return status;
}

void slave_send_current_time(double start_time) {
    double elapsed_time = MPI_Wtime() - start_time;
    MPI_Send(&elapsed_time, 1, MPI_DOUBLE, MASTER_RANK, TIME_GATHER_TAG, MPI_COMM_WORLD);
}
