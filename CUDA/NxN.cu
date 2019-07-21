
// CSC 391
// September 30, 2015
// Project 2

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#define TILE_WIDTH 32

__global__ void matrixMult(float *A, float *B, float *C, int width);
//dummy function
__global__ void first_call();
void fill_matrices(float *A_Matrix, float *B_Matrix, int N);
void print_matrices(float *A_Matrix, float*B_Matrix, float*ANS_Matrix, int N);
int round_gaurav(double number);
void output_file(float* ANS_Matrix, int N);
void check_arguments( int argc, char *argv[] );

//dummy function
__global__ void first_call(){
	int z = 1;
	if (z!=1 ){
	}
}

__global__ void matrixMult(float *A, float *B, float *C, int width) {
	int k = 0;
 	float sum = 0;

	int col = blockDim.x * blockIdx.x + threadIdx.x;
 	int row = blockDim.y * blockIdx.y + threadIdx.y;

	if(col < width && row < width) {
		for (k = 0; k < width; k++)
			sum += A[row * width + k] * B[k * width + col];
		C[row * width + col] = sum;
	}
}
//to print matrices for debugging
void print_matrices(float *A_Matrix, float*B_Matrix, float*ANS_Matrix, int N){
	int i;
	int j;
	printf("A Matrix is: \n");
	// int w = 0;
	for (i = 0; i <  N; i++){
		for (j = 0; j < N; j++){
			printf("%.4f ", *(A_Matrix + i*N + j));
		}
		printf("\n");
	}

	printf("B Matrix is: \n");
	for (i = 0; i <  N; i++){
		for (j = 0; j < N; j++){
			printf("%.4f ", *(B_Matrix + i*N + j));
		}
		printf("\n");
	}

	printf("Resulting Matrix is: \n");
	for (i = 0; i <  N; i++){
		for (j = 0; j < N; j++){
			printf("%.4f ", *(ANS_Matrix + i*N + j));
		}
		printf("\n");
	}
}
//to fill matrices with random floats
void fill_matrices(float *A_Matrix, float *B_Matrix, int N){
	int i;
	int j;

	for (i = 0; i <  N; i++){
		for (j = 0; j < N; j++){

			*(A_Matrix + i*N + j) = (rand()%10)+1;
		}
	}
	for (i = 0; i <  N; i++){
		for (j = 0; j < N; j++){

			*(B_Matrix + i*N + j) = (rand()%10)+1;
		}
	}
}

void output_file(float* ANS_Matrix, int N){
	//open file for writing
	FILE *file_output = fopen("product.txt", "w");
	if (file_output == NULL) {
	    printf("File could not be created. ");
	    exit(1);
	}
	int i;
	int j;
	for (i = 0; i <  N; i++){
		for (j = 0; j < N; j++){
			//outputs the product.dat file
			fprintf(file_output, "%.4f\t", *(ANS_Matrix + i*N + j));
		}
		fprintf(file_output, "\n");
	}
	fclose ( file_output );
}

//below is to round floating point numbers incase the input argument is a float point number
int round_gaurav(double number)
{
    return (number >= 0) ? (int)(number + 0.5) : (int)(number - 0.5);
}



int main ( int argc, char *argv[] )
{


	//treat each number as a float
	//round then to the nearest integer
	// ex 5.4 = 5.0
	// ex 5.6 = 6.0
	//named round_gaurav because it avoids warning error of conflict types with built-in function
	int N = atof(argv[1]);
  int B = atof(argv[2]);
  int T = atof(argv[3]);

	//printf("T = %d\n",N, B);

	//Create the matrices
	//below was in class given information
	float *A_Matrix = (float *)malloc(N * N * sizeof(float));
	float *B_Matrix = (float *)malloc(N * N * sizeof(float));

	//fill matrices with random floats
	fill_matrices(A_Matrix, B_Matrix, N);

	//answer matrix
	float *ANS_Matrix =  (float *)malloc(N * N * sizeof(float));

	//GPU Varibles
	float *dev_A_Matrix;
	float *dev_B_Matrix;
	float *dev_ANS_Matrix;

	//for cuda functions needed
	int size = N * N * sizeof(float);

	//allocate memory for variables
	cudaMalloc((void**)&dev_A_Matrix, size);
	cudaMalloc((void**)&dev_B_Matrix, size);
	cudaMalloc((void**)&dev_ANS_Matrix, size);

	//transfer data to the GPU for calculations
	cudaMemcpy(dev_A_Matrix, A_Matrix, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B_Matrix, B_Matrix, size, cudaMemcpyHostToDevice);

	//the number of blocks is dependent on the tile width
	dim3 dimBlock(B, B);
	//the number of threads
	//need to add plus 1 because need some calculations are not completed then
	dim3 dimGrid(T,T);//(int)ceil(N/dimBlock.x) + 1, (int)ceil(N/dimBlock.y) + 1);

	//dummy call suggested by Cho
	first_call<<<1,1>>>();

	//from in class notes
	clock_t start; // for starting
	clock_t stop; //for stoping
	//total time
	double execution_time;

	//ready set go
	start = clock();

	//do matrix multiple
	matrixMult<<<T, B>>>(dev_A_Matrix, dev_B_Matrix, dev_ANS_Matrix, N);

	//wait for all calculations to finish
	cudaDeviceSynchronize();

	//STOP
	stop = clock();

	//get the important data
	cudaMemcpy(ANS_Matrix, dev_ANS_Matrix, size, cudaMemcpyDeviceToHost);

	//free the variables
	cudaFree(dev_A_Matrix);
	cudaFree(dev_B_Matrix);
	cudaFree(dev_ANS_Matrix);

	//get the execution time
	execution_time = ((double) (stop - start)) / CLOCKS_PER_SEC;
	//Print the execution time
	printf("Execution Time in Seconds: %.8lf\n", execution_time );

	//debugging purposes
	//print_matrices(A_Matrix,B_Matrix,ANS_Matrix, N);

	output_file(ANS_Matrix, N);
}
