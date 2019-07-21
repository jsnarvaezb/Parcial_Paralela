#include <stdio.h>
int N;

__global__ void matrixMultGPU(float *a, float *b, float *c,int N) {
 int k, fil, sum = 0;
 int col = threadIdx.x + blockDim.x * blockIdx.x;
 //int fil = threadIdx.y + blockDim.y * blockIdx.y;

 for(fil=0, fil<N; fil++){
   for (k = 0; k < N; k++) {
    sum += a[fil * N + k] * b[k * N + col];
   }
   c[fil * N + col] = sum;
 }

  //if (col < N && fil < N) {
  //for (k = 0; k < N; k++) {
   //sum += a[fil * N + k] * b[k * N + col];
  //}
  //c[fil * N + col] = sum;
// }
}

int main(int argc, char const *argv[]) {
 N = atoi(argv[1]);
 int blockSize, gridSize;
 gridSize=atoi(argv[2]);
 blockSize= atoi(argv[3]);

 float *c = (float *)malloc(N * N * sizeof(float));
 float *a = (float *)malloc(N * N * sizeof(float));
 float *b = (float *)malloc(N * N * sizeof(float));

 float *dev_a, *dev_b, *dev_c;
 int cont,i,j;

  /* inicializando variables con datos foo*/
 for (i = 0; i < N; i++) {
  for (j = 0; j < N; j++) {
   a[i * N + j] = (rand()%10);
   b[i * N + j] = (rand()%10);
  }
 }

  int size = N * N * sizeof(int);

 cudaMalloc((void **) &dev_a, size);
 cudaMalloc((void **) &dev_b, size);
 cudaMalloc((void **) &dev_c, size);

 cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
 cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

  //dim3 dimGrid(1, 1);
  //dim3 dimBlock(N, N);


  //blockSize = 1024;
  //gridSize = (int)ceil((float)N/blockSize);

  matrixMultGPU<<<gridSize, blockSize>>>(dev_a, dev_b, dev_c, N);

  cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

  cudaFree(dev_a);
 cudaFree(dev_b);
 cudaFree(dev_c);

  //imprimiendo

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

  return 0;

}
