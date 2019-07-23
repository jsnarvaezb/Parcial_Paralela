#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>

int main(int argc, char const *argv[])
{
	int i, j;
	int N = atoi(argv[1]);
	int T = atoi(argv[2]);

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
  omp_set_num_threads(1000);
	/* realizar la multiplicación en paralelo */
  #pragma omp parallel num_threads(T)
{
		int i=0;
		int j=0;
		int k=0;
		int suma = 0;

		#pragma omp for
		for(i = 0; i < N; i++)
		{
			for(j = 0; j < N; j++)
			{
				c[i * N + j] = 0;
				for(k = 0; k < N; k++)
				{
					c[i * N + j] += a[i * N + k] * b[k * N + j];
				}
			}
		}
}


  if (argv[3]!=NULL){
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
