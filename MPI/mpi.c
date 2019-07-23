#include <stdio.h>
#include "mpi.h"

MPI_Status status;

main(int argc, char **argv)
{
  int numtasks,taskid,numworkers,source,dest,col,offset,i,j,k;
  int N = atoi(argv[1]);
  struct timeval start, stop;
  int *a = (float*) malloc(sizeof(float)*N*N);
  int *b = (float*) malloc(sizeof(float)*N*N);
  int *c = (float*) malloc(sizeof(float)*N*N);

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

  numworkers = numtasks-1;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Hello from processor %s, rank %d out of %d processors\n",
        processor_name, taskid, numtasks);

  if (taskid == 0) {
    for (i=0; i<N; i++) {
      for (j=0; j<N; j++) {
        a[i][j]= (rand()%10);
        b[i][j]= (rand()%10);
      }
    }

    col = N/numworkers;
    offset = 0;

    for (dest=1; dest<=numworkers; dest++)
    {
      MPI_Send(&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
      MPI_Send(&col, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
      MPI_Send(&a[offset][0], col*N, MPI_DOUBLE,dest,1, MPI_COMM_WORLD);
      MPI_Send(&b, N*N, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
      offset = offset + col;
    }

    for (i=1; i<=numworkers; i++)
    {
      source = i;
      MPI_Recv(&offset, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
      MPI_Recv(&col, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
      MPI_Recv(&c[offset][0], col*N, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, &status);
    }

   /*printf("Matriz A:\n");
   for (i=0; i<N; i++) {
     for (j=0; j<N; j++)
       printf("%3.2f   ", a[i][j]);
     printf ("\n");
   }

   printf("Matriz B:\n");
   for (i=0; i<N; i++) {
     for (j=0; j<N; j++)
       printf("%3.2f   ", b[i][j]);
     printf ("\n");
   }

    printf("Resultado:\n");
    for (i=0; i<N; i++) {
      for (j=0; j<N; j++)
        printf("%3.2f   ", c[i][j]);
      printf ("\n");
    }*/
  }

  if (taskid > 0) {
    source = 0;
    MPI_Recv(&offset, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(&col, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(&a, col*N, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(&b, N*N, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);

    for (k=0; k<N; k++)
      for (i=0; i<col; i++) {
        c[i][k] = 0.0;
        for (j=0; j<N; j++)
          c[i][k] = c[i][k] + a[i][j] * b[j][k];
      }


    MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
    MPI_Send(&col, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
    MPI_Send(&c, col*N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
  }

  MPI_Finalize();
}
