#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>

static const int MASTER_RANK = 0;
static const int POLL_TAG = 1;
static const int TIME_GATHER_TAG = 2;
static const int TIME_SYNC_TAG = 3;

static const char *poll_message = "TIME_POLL";

static double start_time;
static double accumulated_adjustment;

double current_time() {
    return MPI_Wtime() - start_time;
}

void slave_reply_to_poll(int world_rank);

void slave_wait_for_master_poll(int world_rank);

void slave_send_current_time(int world_rank);

void master_synchronize_clocks(int world_size, int world_rank);

double average_time_differences(int world_size, const double *array);

void slave_wait_for_adjustment(int world_rank);

void master_poll_slaves(int world_size, double *master_send_times);

void master_receive_slave_times(int world_size, double *slave_times_received, double *master_receive_times);

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    start_time = MPI_Wtime();

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const int number_of_queries = 5;
    const int sleep_interval_seconds = 5;
    for (int query = 0; query < number_of_queries; ++query) {
        if (world_rank == MASTER_RANK) {
            master_synchronize_clocks(world_size, world_rank);
            sleep(sleep_interval_seconds);
        } else {
            slave_reply_to_poll(world_rank);
            slave_wait_for_adjustment(world_rank);
        }
    }

    printf("Process [%d]: exited normally\n", world_rank);

    MPI_Finalize();
    return 0;
}

void slave_reply_to_poll(int world_rank) {
    slave_wait_for_master_poll(world_rank);
    slave_send_current_time(world_rank);
}

void slave_wait_for_master_poll(int world_rank) {
//    printf("Process [%d]: waiting for master query\n", world_rank);
    const int MAX_RECEIVED_SIZE = 256;
    MPI_Status status;

    char *buffer = malloc(MAX_RECEIVED_SIZE);
    MPI_Recv(buffer, MAX_RECEIVED_SIZE, MPI_CHAR, MASTER_RANK, POLL_TAG, MPI_COMM_WORLD, &status);
    free(buffer);
    buffer = NULL;
}

void slave_send_current_time(int world_rank) {
    double elapsed_time = current_time();
    printf("Process [%d]: sending current time which is %lf seconds to master [%d]\n", world_rank, elapsed_time,
           MASTER_RANK);
    MPI_Send(&elapsed_time, 1, MPI_DOUBLE, MASTER_RANK, TIME_GATHER_TAG, MPI_COMM_WORLD);
}

void slave_wait_for_adjustment(int world_rank) {
    printf("Process [%d]: waiting to receive adjustment from master\n", world_rank);

    double adjustment;
    MPI_Status status;
    MPI_Recv(&adjustment, 1, MPI_DOUBLE, MASTER_RANK, TIME_SYNC_TAG, MPI_COMM_WORLD, &status);

    double time = current_time();
    accumulated_adjustment += adjustment;
    printf("Process [%d]: local time is %.6lf, received adjustment %.6lf, adjusted time is %.6lf\n", world_rank, time + accumulated_adjustment - adjustment, adjustment, time + accumulated_adjustment);
}

void master_synchronize_clocks(int world_size, int world_rank) {
    double *master_send_times = malloc(world_size * sizeof(double));
    double *slave_times_received = malloc(world_size * sizeof(double));
    double *master_receive_times = malloc(world_size * sizeof(double));
    double *master_slave_time_differences = malloc(world_size * sizeof(double));

    master_poll_slaves(world_size, master_send_times);
    master_receive_slave_times(world_size, slave_times_received, master_receive_times);

    for (int rank = 0; rank < world_size; ++rank) {
        if (rank != MASTER_RANK) {
            master_slave_time_differences[rank] = (master_send_times[rank] + master_receive_times[rank]) / 2.0 - slave_times_received[rank];
            printf("Master [%d]: sent to slave [%d] at %lf, slave sent back %lf, master received at %lf, master-slave delta is %lf\n",
                   world_rank,
                   rank,
                   master_send_times[rank],
                   slave_times_received[rank],
                   master_receive_times[rank],
                   master_slave_time_differences[rank]
            );
        }
    }

    master_slave_time_differences[MASTER_RANK] = 0.0;
    double average_delta = average_time_differences(world_size, master_slave_time_differences);
    printf("Master: the average_time_differences time difference is %lf\n", average_delta);

    for (int rank = 0; rank < world_size; ++rank) {
        double adjustment;
        if (rank != MASTER_RANK) {
            adjustment = average_delta - master_slave_time_differences[rank];
            MPI_Send(&adjustment, 1, MPI_DOUBLE, rank, TIME_SYNC_TAG, MPI_COMM_WORLD);
        } else {
            adjustment = average_delta;
            accumulated_adjustment += adjustment;
        }

//        printf("Master [%d]: process [%d] will have be adjusted by %lf\n", MASTER_RANK, rank, adjustment);
    }


    free(master_send_times);
    master_send_times = NULL;

    free(slave_times_received);
    slave_times_received = NULL;

    free(master_receive_times);
    master_receive_times = NULL;

    free(master_slave_time_differences);
    master_slave_time_differences = NULL;
}

void master_receive_slave_times(int world_size, double * const slave_times_received, double * const master_receive_times) {
    MPI_Status status;
    for (int source_rank = 0; source_rank < world_size; ++source_rank) {
        if (source_rank != MASTER_RANK) {
            MPI_Recv(&slave_times_received[source_rank], 1, MPI_DOUBLE, source_rank, TIME_GATHER_TAG, MPI_COMM_WORLD,
                     &status);
            master_receive_times[source_rank] = current_time();
        }
    }
}

void master_poll_slaves(int world_size, double * const master_send_times) {
//    printf("Broadcasting a query for local time\n");
    for (int destination_rank = 0; destination_rank < world_size; ++destination_rank) {
        if (destination_rank != MASTER_RANK) {
            master_send_times[destination_rank] = current_time();
            MPI_Send(poll_message, 1 + strlen(poll_message), MPI_CHAR, destination_rank, POLL_TAG,
                     MPI_COMM_WORLD);
        }
    }
//    printf("Master process: Sent all poll messages\n");
}

double average_time_differences(int world_size, const double *array) {
    double result = 0.0;
    for (int rank = 0; rank < world_size; ++rank) {
        if (rank != MASTER_RANK) {
            result += 1.0 / world_size * array[rank];
        }
    }
    return result;
}
