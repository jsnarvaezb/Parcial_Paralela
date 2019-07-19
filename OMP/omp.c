#include <stdio.h>
#include "omp.h"
#include <time.h>
#include <stdlib.h>

int main(int argc, char const *argv[])
{
	int i, j;
	//srand(time(NULL));
	int n = atoi(argv[1]);
	//int vector1[n][n], vector2[n][n], vectorSalida[n][n];

	float *a = (float*) malloc(sizeof(float)*n*n);
  float *b = (float*) malloc(sizeof(float)*n*n);
  float *c = (float*) malloc(sizeof(float)*n*n);

	for(i = 0; i < n; i++)
	{
		for(j = 0; j < n; j++)
		{
			//vector1[i][j] = (rand()%10)+1;
			a[i * n + j]= (rand()%10)+1;
			b[i * n + j]= (rand()%10)+1;
			//vector2[i][j] = (rand()%10)+1;
		}
	}

	/* realizar la multiplicación en paralelo */
	{
		int i, j, k, suma = 0;

		#pragma omp for

		for(i = 0; i < n; i++)
		{
			for(j = 0; j < n; j++)
			{
				c[i * n + j] = 0;
				for(k = 0; k < n; k++)
				{
					c[i * n + j] += a[i * n + k] * b[k * n + j];
				}
			}
		}
	}

	return 0;
}
