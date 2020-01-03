#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

static const int MASTER_RANK = 0;
static const int POLL_TAG = 1;
static const int TIME_GATHER_TAG = 2;
static const int TIME_SYNC_TAG = 3;

static const char *poll_message = "TIME_POLL";

static double start_time;

double current_time() {
    return MPI_Wtime() - start_time;
}

void slave_reply_to_poll(int world_rank);

void slave_wait_for_master_poll(int world_rank);

void slave_send_current_time(int world_rank);

void master_synchronize_clocks(int world_size, int world_rank);

double average(int number_of_elements, const double *array);

void slave_wait_for_adjustment(int world_rank);

void master_broadcast_poll(int world_size);

void print_array(int number_of_elements, const double *array);

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    start_time = MPI_Wtime();

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == MASTER_RANK) {
        master_broadcast_poll(world_size);
        slave_reply_to_poll(world_rank);

        master_synchronize_clocks(world_size, world_rank);
        slave_wait_for_adjustment(world_rank);
    } else {
        slave_reply_to_poll(world_rank);
        slave_wait_for_adjustment(world_rank);
    }

    printf("Process [%d]: exited normally\n", world_rank);

    MPI_Finalize();
    return 0;
}

void master_broadcast_poll(int world_size) {
    printf("Broadcasting a query for local time\n");
    for (int destination_rank = 0; destination_rank < world_size; ++destination_rank) {
        MPI_Send(poll_message, 1 + strlen(poll_message), MPI_CHAR, destination_rank, POLL_TAG, MPI_COMM_WORLD);
    }

    printf("Master process: Sent all poll messages\n");
}

void slave_reply_to_poll(int world_rank) {
    slave_wait_for_master_poll(world_rank);
    slave_send_current_time(world_rank);
}

void slave_wait_for_adjustment(int world_rank) {
    printf("Process [%d]: waiting to receive delta from master\n", world_rank);

    double delta;
    MPI_Status status;
    MPI_Recv(&delta, 1, MPI_DOUBLE, MASTER_RANK, TIME_SYNC_TAG, MPI_COMM_WORLD, &status);
    printf("Process [%d] received a delta of %.6lf to be added to local time %lf\n", world_rank, delta, current_time());
}

void slave_wait_for_master_poll(int world_rank) {
    printf("Process [%d]: waiting for master query\n", world_rank);
    const int MAX_RECEIVED_SIZE = 256;
    MPI_Status status;

    char *buffer = malloc(MAX_RECEIVED_SIZE);
    MPI_Recv(buffer, MAX_RECEIVED_SIZE, MPI_CHAR, MASTER_RANK, POLL_TAG, MPI_COMM_WORLD, &status);
    free(buffer);
    buffer = NULL;
}

void slave_send_current_time(int world_rank) {
    double elapsed_time = current_time();
    printf("Process [%d]: sending current time which is %lf seconds (from start of program) to master [%d]\n", world_rank, elapsed_time, MASTER_RANK);
    MPI_Send(&elapsed_time, 1, MPI_DOUBLE, MASTER_RANK, TIME_GATHER_TAG, MPI_COMM_WORLD);
}

void master_synchronize_clocks(int world_size, int world_rank) {
    double *times_received = malloc(world_rank * sizeof(double));
    double *round_trip_times = malloc(world_size * sizeof(double));

    MPI_Status status;

    /* TODO: the round trip times will be monotonically increasing (if msg #1 block but #2 would have been
     * received beforehand) but the times would be #1 < #2, which may lead to incorrect results (depends on the
     * desired precision)
     */
    for (int source_rank = 0; source_rank < world_size; ++source_rank) {
        MPI_Recv(&times_received[source_rank], 1, MPI_DOUBLE, source_rank, TIME_GATHER_TAG, MPI_COMM_WORLD, &status);
        printf("Process [%d]: Received time %lf\n", world_rank, times_received[source_rank]);

        round_trip_times[source_rank] = current_time() - times_received[source_rank];
    }

    printf("Times received: ");
    print_array(world_size, times_received);

    printf("Round trip times: ");
    print_array(world_size, round_trip_times);

    double correct_time = average(world_size, times_received);
    printf("Master [%d]: The correct time was %lf\n", MASTER_RANK, correct_time);

    for (int dest_rank = 0; dest_rank < world_size; ++dest_rank) {
        double delta = correct_time - times_received[dest_rank] + round_trip_times[dest_rank];
        MPI_Send(&delta, 1, MPI_DOUBLE, dest_rank, TIME_SYNC_TAG, MPI_COMM_WORLD);
    }

    free(times_received);
    times_received = NULL;

    free(round_trip_times);
    round_trip_times = NULL;
}

double average(int number_of_elements, const double *array) {
    double result = 0.0;
    for (int i = 0; i < number_of_elements; ++i) {
        result += 1.0 / number_of_elements * array[i];
    }
    return result;
}

void print_array(int number_of_elements, const double *array) {
    for (int i = 0; i < number_of_elements; ++i) {
        printf("%10.6lf ", array[i]);
    }
    printf("\n");
}
