#include <bits/stdc++.h>
#include <ctime>
#include <ratio>
#include <chrono>
using namespace std;
#define TILE_DIM 16
#define CONV_THRESHOLD 1e-3

bool check_convergence(int N, double* D, double* D_new){
	double sqr_diff = 0;
	for(int i=0; i<N; i++){
		double diff = D_new[i*N + i] - D[i*N + i];
		if(diff<0) sqr_diff -= diff;
		else sqr_diff += diff;
		// if(sqr_diff > CONV_THRESHOLD) return false;
	}
	cout << sqr_diff << endl;
	return (sqr_diff < CONV_THRESHOLD);
}

double check_eigenvals(int N, double* D){
	double out = 0;
	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){
			if(i!=j){
				out+=fabs(D[i*N + j]);
			}
		}
	} 
	return out;
}
__global__ void pq_change(int N, int *p, int*q){
	// printf("Updating PQ\n");
	int tid = threadIdx.x;
	int i = blockIdx.x;

	int ind1 = (tid + i)%(N-1);
   	int ind2;
   	if(tid != 0) ind2 = ((N-tid)+i - 1)%(N-1);
   	else ind2 = N - 1;

   	int valp = min(ind1, ind2);
   	int valq = max(ind1, ind2);

   	// printf("%d %d\n", valp, valq);

   	p[i*(N)/2 + tid] = valp;
   	q[i*(N)/2 + tid] = valq;
}

__global__ void cosandsin(int *N, double *D, double *c, double *s, int *pcurr, int *qcurr){
	// printf("Inside cossin\n");
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	// printf("%d\n", tid);
	int row = pcurr[tid];
	int col = qcurr[tid];

	// printf("[@] Inside cossin update: \n" );

	// printf("[@] row col: %d %double\n\n", row, col);

	double p = D[row*(*N) + col];
    double y = (D[col*(*N) + col] - D[row*(*N) + row]) / 2.0;
    double d = fabs(y) + sqrt(p*p + y*y);
    double r = sqrt(p*p + d*d);

    // printf("%f %f %f %f\n", p, y, d, r);

    if(fabs(p) < CONV_THRESHOLD && fabs(d) < CONV_THRESHOLD){
    	c[tid] = 1.0;
    	s[tid] = 0.0;
    }
    else{
	    c[tid] = d / r;
		s[tid] = (fabs(y)/y)*(p / r);
    }

    // printf("%d: %f %f\n\n\n\n",tid, (double)c[tid], (double)s[tid] );
}


__global__ void rotate_rows(int* N, double* D, double* out, double* c, double* s, int* pcurr, int* qcurr){
	__shared__ int p, q;
	__shared__ double co, si;

	if(threadIdx.x == 0){
		p = pcurr[blockIdx.x];
		q = qcurr[blockIdx.x];
		co = c[blockIdx.x];
		si = s[blockIdx.x];
		// printf("[@] Inside Row update: \n" );
		// printf("%d %d %f %f \n", p, q, co, si);
	
	}



	__syncthreads();



	int i = threadIdx.x;


	double val1 = D[p*(*N)+i];
	double val2 = D[q*(*N)+i];



	out[i*(*N)+p] = co*val1 - si*val2;
	// out[i*(*N)+q] = si*val1 + co*val2;

}

__global__ void rotate_rows2(int* N, double* D, double* out, double* c, double* s, int* pcurr, int* qcurr){
	__shared__ int p, q;
	__shared__ double co, si;

	if(threadIdx.x == 0){
		p = pcurr[blockIdx.x];
		q = qcurr[blockIdx.x];
		co = c[blockIdx.x];
		si = s[blockIdx.x];
	}
	__syncthreads();

	int i = threadIdx.x;

	double val1 = D[p*(*N)+i];
	double val2 = D[q*(*N)+i];

	// out[i*(*N)+p] = co*val1 - si*val2;
	out[i*(*N)+q] = si*val1 + co*val2;

}

__global__ void rotate_cols(int* N, double* D, double* out, double* c, double* s, int* pcurr, int* qcurr){
	__shared__ int p, q;
	__shared__ double co, si;

	if(threadIdx.x == 0){
		p = pcurr[blockIdx.x];
		q = qcurr[blockIdx.x];
		co = c[blockIdx.x];
		si = s[blockIdx.x];
	}
	__syncthreads();
	int i = threadIdx.x;

	double val1 = D[p*(*N)+i];
	double val2 = D[q*(*N)+i];

	// double val3 = D1[i*(*N)+p];
	// double val4 = D1[i*(*N)+q];


	out[p*(*N)+i] = co*val1 - si*val2;
	out[q*(*N)+i] = si*val1 + co*val2;

	// out1[i*(*N)+p] = co*val3 - si*val4;
	// out1[i*(*N)+q] = si*val3 + co*val4;


}


__global__ void get_ev(double* old_arr, double* new_arr){
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	new_arr[tid] = old_arr[tid];
}


void jacobi_parallel(int N, double* D, double* eigenvecs_out, double* eigenvals_out){
	double *eigenvals = eigenvals_out;
	double *eigenvecs = eigenvecs_out;
	int N2 = N;
	double *ENEW;
	double *DNEW;
	if(N%2==1){

		ENEW = (double*)calloc((N+1)*(N+1), sizeof(double));
		DNEW = (double*)calloc((N+1)*(N+1), sizeof(double));
		for(int i=0; i<N; i++){
			for(int j=0; j<N; j++){
				DNEW[i*(N+1) + j] = D[i*N + j];
				ENEW[i*(N+1) + j] = 0;
			}
			ENEW[i*(N+1)+i] = 1;
		}
		D = DNEW;
		N = N+1;
		D[N*N - 1] = 1;
		ENEW[N*N - 1] = 1;
		eigenvals = D;
		eigenvecs = ENEW;
		// printf("Done\n");
	}

	double *dD, *Dtemp, *eignevecs_D, *eignevecs_D_temp;
	std::chrono::high_resolution_clock::time_point t1, t2;
	t1 = std::chrono::high_resolution_clock::now();
	cudaMalloc((void**)&dD, sizeof(double)*N*N);
	cudaMalloc((void**)&Dtemp, sizeof(double)*N*N);
	cudaMalloc((void**)&eignevecs_D, sizeof(double)*N*N);
	cudaMalloc((void**)&eignevecs_D_temp, sizeof(double)*N*N);


	// for(int i=0; i<N*N; i++){
	// 	eigenvals[i] = D[i];
	// }

	cudaMemcpy(dD, D, sizeof(double)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(Dtemp, D, sizeof(double)*N*N, cudaMemcpyHostToDevice);
	
	cudaMemcpy(eignevecs_D, eigenvecs, sizeof(double)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(eignevecs_D_temp, eigenvecs, sizeof(double)*N*N, cudaMemcpyHostToDevice);

	int *dN, *dp, *dq;
	double *c, *s;
	cudaMalloc((void **)&dN, sizeof(int));
	// cudaMalloc((void **)&di, sizeof(int));
	cudaMalloc((void **)&c, sizeof(double)*N/2);
	cudaMalloc((void **)&s, sizeof(double)*N/2);

	// print_matrix("D", N, N, D);

	cudaMemcpy(dN, &N, sizeof(int), cudaMemcpyHostToDevice);

	double *Dvoidtemp = (double*)malloc(sizeof(double)*N*N);
	double conv = false;
	t2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

  	// std::cout << "[@]Initial Jacobi time: " << time_span.count() << " seconds.\n";

	cudaMalloc((void **)&dp, sizeof(int)*N*(N-1)/2);
	cudaMalloc((void **)&dq, sizeof(int)*N*(N-1)/2);
	pq_change<<<N-1, N/2>>>(N, dp, dq);


	int *p = (int*)malloc(sizeof(int)*N*(N-1)/2);
	int *q = (int*)malloc(sizeof(int)*N*(N-1)/2);

	cudaMemcpy(p, dp, sizeof(int)*N*(N-1)/2, cudaMemcpyDeviceToHost);
	cudaMemcpy(q, dq, sizeof(int)*N*(N-1)/2, cudaMemcpyDeviceToHost);
	

	cudaDeviceSynchronize(); 
	// print_matrix("D", 4, 4, D);
	int sweeps = 0;
	while(!conv){
		t1 = std::chrono::high_resolution_clock::now();

		int N1;

		if(N%2 != 0) N1 = N;
		else N1 = N-1;
		// if(N%2 != 0) N2 = N-1;
		// else N2 = N;
		for(int i=0; i<N1; i++){
			int *currp = dp+(i*(N/2));
			int *currq = dq+(i*(N/2));
			// cudaMemcpy(di, &i, sizeof(int), cudaMemcpyHostToDevice);

			// cudaMemcpy(eigenvals, dD, sizeof(double)*N*N, cudaMemcpyDeviceToHost);
			// print_matrix("eigenvals", N, N, eigenvals);
			// printf("Printing finished\n");

			cossin<<<N/2, 1>>>(dN, dD, c, s, currp, currq);
			cudaDeviceSynchronize();
			// printf("cosandsin updated\n");
			rotate_rows<<<N/2, N>>>(dN, dD, Dtemp, c, s, currp, currq);
			rotate_rows2<<<N/2, N>>>(dN, dD, Dtemp, c, s, currp, currq);
			
			// cudaMemcpy(eigenvals, Dtemp, sizeof(double)*N*N, cudaMemcpyDeviceToHost);
			// print_matrix("After row update", N, N, eigenvals);

			cudaDeviceSynchronize();
		    // dim3 dimBlock(N, 2);

			rotate_cols<<<N/2, N>>>(dN, Dtemp, dD, c, s, currp, currq);
			
			// cudaMemcpy(eigenvals, dD, sizeof(double)*N*N, cudaMemcpyDeviceToHost);
			// print_matrix("After col update", N, N, eigenvals);

			// cudaDeviceSynchronize();
			rotate_cols<<<N/2, N>>>(dN, eignevecs_D, eignevecs_D_temp, c, s, currp, currq);
			cudaDeviceSynchronize();
			get_ev<<<N, N>>>(eignevecs_D_temp, eignevecs_D);
			cudaDeviceSynchronize();
			// while();

		}
		// while(true){};

		cudaMemcpy(Dvoidtemp, dD, sizeof(double)*N*N, cudaMemcpyDeviceToHost);

		// print_matrix("D", 4, 4, D);

		cout << "Sweep " << ++sweeps << ": ";

		conv = check_convergence(N, eigenvals, Dvoidtemp);
		t2 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);


		double* tempor = eigenvals;
		eigenvals = Dvoidtemp;
		Dvoidtemp = tempor;

		// dD = Dtemp2;
		double valdiff = check_eigenvals(N, eigenvals);
		// cout << "Valdiff: " << valdiff << endl;
		// std::cout << "[@]Time taken: " << time_span.count() << " seconds.\n";

	}
	// print_matrix("Eigenvals", N, N, eigenvals);
	double* eigenvecs_temp = (double*)malloc(sizeof(double)*N*N);
	cudaMemcpy(eigenvecs_temp, eignevecs_D, sizeof(double)*N*N, cudaMemcpyDeviceToHost);
	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){
			eigenvecs[j*N+i] = eigenvecs_temp[N*i + j];
		}
	}

	if(N2%2 == 1){
		for(int i=0; i<N-1; i++){
			for(int j=0; j<N-1; j++){
				eigenvecs_out[i*(N-1) + j] = eigenvecs[i*N + j];
				eigenvals_out[i*(N-1) + j] = eigenvals[i*N + j];

			}
		}
		free(eigenvals);
		free(eigenvecs);
	}

	// free(eigenvecs);
	// free(eigenvals);
	free(eigenvecs_temp);

	cudaFree(dD);
	cudaFree(Dtemp);
	cudaFree(dN);
	cudaFree(c);
	cudaFree(s);
	cudaFree(dp);
	// cudaFree(di)
	cudaFree(dq);
	cudaFree(eignevecs_D);
	cudaFree(eignevecs_D_temp);

	// free(Dvoidtemp);


	// print_matrix("D", N, N, D);
	// eigenvals = D;
}
int main(){
	ofstream ofile;
	ofile.open("output.txt");
	cudaEvent_t start,end;
	for(int size=2;size<=512;size*=2){
		cudaEventCreate(&start);
        cudaEventCreate(&end);	
		fstream infile;
		infile.open("input_"+to_string(size)+".txt");
		int N;
		infile>>N;
		cudaEventRecord(start);
		double* D=(double*)calloc(N*N, sizeof(double));
		double data;

		double* temp = (double*)calloc(N*N, sizeof(double));
		double* E = (double*)calloc(N*N, sizeof(double));
		
		for(int i=0; i<N; i++){
			E[i*N+i] = 1;
		}
		for(int i=0; i<N; i++){
			infile>>data;
			D[i*N+i] = data;
		}
		
		jacobi_parallel(N,D,E,temp);

		infile.close();

		cudaThreadSynchronize();
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        float milliseconds=0;
		cudaEventElapsedTime(&milliseconds,start,end);
		ofile<<size<<":"<<milliseconds*1000<<"\n";
		}
	
		ofile.close();
		return 0;
	
	
}