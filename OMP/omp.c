#include <stdio.h>
#include "omp.h"
#include <time.h>
#include <stdlib.h>

int main(int argc, char const *argv[])
{
	int i, j;
	int N = atoi(argv[1]);

	float *a = (float*) malloc(sizeof(float)*N*N);
  float *b = (float*) malloc(sizeof(float)*N*N);
  float *c = (float*) malloc(sizeof(float)*N*N);

	for(i = 0; i < N; i++)
	{
		for(j = 0; j < N; j++)
		{
			a[i * N + j]= (rand()%10);
			b[i * N + j]= (rand()%10);
		}
	}

	/* realizar la multiplicación en paralelo */

	{
		int i, j, k, suma = 0;

		#pragma omp parallel for

		for(i = 0; i < N; i++)
		{
			#pragma omp parallel for
			for(j = 0; j < N; j++)
			{
				c[i * N + j] = 0;
				#pragma omp parallel for
				for(k = 0; k < N; k++)
				{
					c[i * N + j] += a[i * N + k] * b[k * N + j];
				}
			}
		}
	}
  if (argv[2]!=NULL){
	printf("Matrix A --------------------\n");
  for (int y = 0; y < N; y++) {
   for (int x = 0; x < N; x++) {

    printf("%f ", a[y * N + x]);
   }
   printf("\n");
  }
  printf("Matrix B --------------------\n");
  for (int y = 0; y < N; y++) {
   for (int x = 0; x < N; x++) {
    printf("%f ", b[y * N + x]);
   }
   printf("\n");
  }
  printf("Matrix C --------------------\n");

 for (int y = 0; y < N; y++) {
  for (int x = 0; x < N; x++) {
   printf("%f ", c[y * N + x]);
  }
  printf("\n");
 }
}
	return 0;
}
