#include "config.h"

#ifdef __NVCC__
	#pragma diag_suppress 20012
#endif

#pragma warning(disable: 4819)


#define WARPSIZE	32

#include <glm/glm.hpp>
#include <ctime>
#include <vector>


#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

using namespace glm;

constexpr int order = 20;

constexpr int node_res = 1;

std::vector<float> velocity_x(res*res*res, 0);
std::vector<float> velocity_y(res*res*res, 0);
std::vector<float> velocity_z(res*res*res, 0);
std::vector<float> node_velocity_x(res*res*res, 0.8);
std::vector<float> node_velocity_y(res*res*res, 0.5);
std::vector<float> node_velocity_z(res*res*res, 0.5);


constexpr float fit_step_size = 4e-6f;
constexpr float sim_step_size = 1e-2f;


inline __host__ __device__ float get_coord(int index) { return 0.99f * (2.0f * index / res - 1.0f); }
inline __host__ __device__ int IDX_o2(int x, int y) { return y + order * x; }
inline __host__ __device__ int IDX_o(int x, int y, int z) { return z + order * (y + order * x); }
inline __host__ __device__ int IDX_r(int x, int y, int z) { return z + res * (y + res * x); }



static float *d_legendre;
static float *d_elements, *d_Ppq, *d_advect_legendre_terms;
static float *d_a, *d_b, *d_c;
static float *d_a2, *d_b2, *d_c2;
static float *d_velocity_x, *d_velocity_y, *d_velocity_z;
static float *d_node_velocity_x, *d_node_velocity_y, *d_node_velocity_z;
static float *d_velocity_mse, *d_velocity_mse_total;


std::vector<float>& getVelocityX() { return velocity_x; }
std::vector<float>& getVelocityY() { return velocity_y; }
std::vector<float>& getVelocityZ() { return velocity_z; }
std::vector<float>& getNodeVelocityX() { return node_velocity_x; }
std::vector<float>& getNodeVelocityY() { return node_velocity_y; }
std::vector<float>& getNodeVelocityZ() { return node_velocity_z; }

void cudaError(const char* msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
		printf("cudaErr(%d) in %s: %s \n", err, msg, cudaGetErrorString(err));
}

void init()
{
	srand(time(0));

	// insert source fluid
#pragma omp parallel for
	for (int i = 0; i < res; i++)
	{
		float x = get_coord(i);
		for (int j = 0; j < res; j++)
		{
			float y = get_coord(j);
			for (int k = 0; k < res; k++)
			{
				float z = get_coord(k);

				int ijk_r = IDX_r(i, j, k);
				velocity_x[ijk_r] = -2*x*y*z + 0.01f;
				velocity_y[ijk_r] = 2*y*y*z + 0.01f;
				velocity_z[ijk_r] = z*z*y + 0.01f;
				//if (x > 0)
				//	velocity_x[ijk_r] = 0;
				//else
				//	velocity_x[ijk_r] = -x;
				//velocity_y[ijk_r] = 0;
				//velocity_z[ijk_r] = 0;
			}
		}
	}

	std::vector<float*> legendre(res, NULL);
#pragma omp parallel for
	for (int i = 0; i < res; i++)
	{
		float z = get_coord(i);

		double* legendre_double_arr = new double[order];
		legendre_double_arr[0] = 1.0;
		legendre_double_arr[1] = z;
		for (int o = 2; o < order; o++)
			legendre_double_arr[o] = ((2.0 * o - 1.0) * z * legendre_double_arr[o - 1] - (o - 1.0) * legendre_double_arr[o - 2]) / o;
		float* legendre_arr = new float[order];
		for (int o = 0; o < order; o++)
			legendre_arr[o] = (float)legendre_double_arr[o];
		delete[] legendre_double_arr;
		legendre[i] = legendre_arr;
	}

	std::vector<double> A(2 * order, 0.0f);
	A[0] = 1.0f;
	for (int o = 1; o < 2 * order; o++)
		A[o] = A[o - 1] * (2.0 * o - 1.0) / o;

	std::vector<std::vector<float*>> Ppq(order, std::vector<float*>(order, NULL));
#pragma omp parallel for
	for (int pq = 0; pq < order * order; pq++)
	{
		int p = pq % order;
		int q = pq / order;

		int min_pq = min(p, q);

		float* Ppq_arr = new float[order];
		memset(Ppq_arr, 0, sizeof(float) * order);

		for (int r = 0; r <= min_pq; r++)
		{
			double c = 1.0 - 2.0 * r / (2.0 * (p + q - r) + 1.0);
			c *= A[p - r] * A[r] * A[q - r] / A[p + q - r];
			if(p + q - 2 * r < order)
				Ppq_arr[p + q - 2 * r] = (float)c;
		}
		Ppq[p][q] = Ppq_arr;
	}

	std::vector<std::vector<std::vector<float>>> elements(
		order, std::vector<std::vector<float>>(order, std::vector<float>(order, 0)));
#pragma omp parallel for
	for (int a = 0; a < order; a++)
		for (int b = 0; b < order; b++)
			for (int o = 0; o < order; o++)
			{
				double sum = 0.0;
				for (int i = a - 1; i >= 0; i -= 2)
					sum += Ppq[i][b][o] / (2.0 * i + 1.0);
				elements[a][b][o] = (float)sum;
			}

//	std::vector<std::vector<std::vector<float>>> advect_legendre_terms(
//		order, std::vector<std::vector<float>>(order, std::vector<float>(2*order, 0)));
//#pragma omp parallel for
//	for (int ab = 0; ab < 2*order; ab++)
//		for (int cd = 0; cd < 2*order; cd++)
//			for (int ef = 0; ef < 2 * order; ef++)
//			{
//				advect_legendre_terms[ab][cd][ef] = 0;
//			}


	cudaMalloc(&d_legendre, sizeof(float) * res * order);
	cudaMalloc(&d_velocity_x, sizeof(float) * res * res * res);
	cudaMalloc(&d_velocity_y, sizeof(float) * res * res * res);
	cudaMalloc(&d_velocity_z, sizeof(float) * res * res * res);
	cudaMalloc(&d_node_velocity_x, sizeof(float) * res * res * res);
	cudaMalloc(&d_node_velocity_y, sizeof(float) * res * res * res);
	cudaMalloc(&d_node_velocity_z, sizeof(float) * res * res * res);
	cudaMalloc(&d_velocity_mse, sizeof(float) * order * order * order);
	cudaMalloc(&d_velocity_mse_total, sizeof(float) * 1);

	cudaMalloc(&d_Ppq, sizeof(float) * order * order * order);
	cudaMalloc(&d_elements, sizeof(float) * order * order * order);
	cudaMalloc(&d_advect_legendre_terms, sizeof(float) * order * order * order);
	cudaMalloc(&d_a, sizeof(float) * order * order * order);
	cudaMalloc(&d_b, sizeof(float) * order * order * order);
	cudaMalloc(&d_c, sizeof(float) * order * order * order);
	cudaMalloc(&d_a2, sizeof(float) * order * order * order);
	cudaMalloc(&d_b2, sizeof(float) * order * order * order);
	cudaMalloc(&d_c2, sizeof(float) * order * order * order);

	cudaMemsetAsync(d_node_velocity_x, 0, sizeof(float) * res * res * res);
	cudaMemsetAsync(d_node_velocity_y, 0, sizeof(float) * res * res * res);
	cudaMemsetAsync(d_node_velocity_z, 0, sizeof(float) * res * res * res);
	cudaMemsetAsync(d_a, 0, sizeof(float) * order * order * order);
	cudaMemsetAsync(d_b, 0, sizeof(float) * order * order * order);
	cudaMemsetAsync(d_c, 0, sizeof(float) * order * order * order);
	cudaMemsetAsync(d_a2, 0, sizeof(float) * order * order * order);
	cudaMemsetAsync(d_b2, 0, sizeof(float) * order * order * order);
	cudaMemsetAsync(d_c2, 0, sizeof(float) * order * order * order);

	cudaMemcpyAsync(d_velocity_x, velocity_x.data(), sizeof(float) * res * res * res, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_velocity_y, velocity_y.data(), sizeof(float) * res * res * res, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_velocity_z, velocity_z.data(), sizeof(float) * res * res * res, cudaMemcpyHostToDevice);

	int offset = 0;
	for (int r = 0; r < res; r++)
	{
		cudaMemcpyAsync(d_legendre + offset, legendre[r], sizeof(float) * order, cudaMemcpyHostToDevice);
		offset += order;
	}
	offset = 0;
	for (int a = 0; a < order; a++)
	{
		for (int b = 0; b < order; b++)
		{
			cudaMemcpyAsync(d_Ppq + offset, Ppq[a][b], sizeof(float) * order, cudaMemcpyHostToDevice);
			cudaMemcpyAsync(d_elements + offset, elements[a][b].data(), sizeof(float) * order, cudaMemcpyHostToDevice);
			offset += order;
		}
	}
//	offset = 0;
//	for (int ab = 0; ab < 2*order; ab++)
//	{
//		for (int cd = 0; cd < 2*order; cd++)
//		{
//			cudaMemcpyAsync(
//				d_advect_legendre_terms + offset, 
//				advect_legendre_terms[ab][cd].data(), 
//				sizeof(float) * 2 * order, cudaMemcpyHostToDevice);
//			offset += 2*order;
//		}
//	}
	cudaDeviceSynchronize();
}

bool is_fit = false;

__global__ void cuda_compute_node_velocity(
	const float* __restrict__ d_legendre,
	const float* __restrict__ d_a,
	const float* __restrict__ d_b,
	const float* __restrict__ d_c,
	float* __restrict__ d_node_velocity_x,
	float* __restrict__ d_node_velocity_y,
	float* __restrict__ d_node_velocity_z
)
{
	volatile int i = threadIdx.x + blockIdx.x * blockDim.x;
	volatile int j = threadIdx.y + blockIdx.y * blockDim.y;
	volatile int k = threadIdx.z + blockIdx.z * blockDim.z;
	if (i >= res || j >= res || k >= res)
		return;

	float cf_x = 0.0f;
	float cf_y = 0.0f;
	float cf_z = 0.0f;
	for (int o = 0; o < order; o++)
	{
		float leg_x = d_legendre[i * order + o];
		for (int p = 0; p < order; p++)
		{
			float leg_xy = leg_x * d_legendre[j * order + p];
			for (int q = 0; q < order; q++)
			{
				float leg_xyz = leg_xy * d_legendre[k * order + q];

				int idx_o = IDX_o(o, p, q);
				cf_x += d_a[idx_o] * leg_xyz;
				cf_y += d_b[idx_o] * leg_xyz;
				cf_z += d_c[idx_o] * leg_xyz;
			}
		}
	}
//	printf("%.3f ", cf_x);
	int idx_r = IDX_r(i, j, k);
	d_node_velocity_x[idx_r] = cf_x;
	d_node_velocity_y[idx_r] = cf_y;
	d_node_velocity_z[idx_r] = cf_z;
}

__global__ void cuda_fit(
	const float* __restrict__ d_legendre,
	const float* __restrict__ d_velocity_x,
	const float* __restrict__ d_velocity_y,
	const float* __restrict__ d_velocity_z,
	const float* __restrict__ d_node_velocity_x,
	const float* __restrict__ d_node_velocity_y,
	const float* __restrict__ d_node_velocity_z,
	float* __restrict__ d_a,
	float* __restrict__ d_b,
	float* __restrict__ d_c,
	float* __restrict__ d_velocity_mse
)
{
	volatile int o = threadIdx.x + blockIdx.x * blockDim.x;
	volatile int p = threadIdx.y + blockIdx.y * blockDim.y;
	volatile int q = threadIdx.z + blockIdx.z * blockDim.z;
	if (o >= order || p >= order || q >= order)
		return;

	float a_op = 0.0f;
	float b_op = 0.0f;
	float c_op = 0.0f;
	float rse = 0.0f;
	for (int i = 0; i < res; i++)
	{
		float leg_x = d_legendre[i * order + o];
		for (int j = 0; j < res; j++)
		{
			float leg_xy = leg_x * d_legendre[j * order + p];
			for (int k = 0; k < res; k++)
			{
				float leg_xyz = leg_xy * d_legendre[k * order + q];

				int idx_r = IDX_r(i, j, k);
				float x_diff = (d_velocity_x[idx_r] - d_node_velocity_x[idx_r]);
				float y_diff = (d_velocity_y[idx_r] - d_node_velocity_y[idx_r]);
				float z_diff = (d_velocity_z[idx_r] - d_node_velocity_z[idx_r]);
				rse += x_diff * x_diff + y_diff * y_diff + z_diff * z_diff;
				a_op += leg_xyz * x_diff;
				b_op += leg_xyz * y_diff;
				c_op += leg_xyz * z_diff;
			}
		}
	}
	int idx_o = IDX_o(o, p, q);
	d_a[idx_o] += a_op * fit_step_size;
	d_b[idx_o] += b_op * fit_step_size;
	d_c[idx_o] += c_op * fit_step_size;
	d_velocity_mse[idx_o] = rse;
}

__global__ void reduce_ws(float* gdata, float* out)
{
	__shared__ float sdata[32];
	int tid = threadIdx.x;
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	float val = 0.0f;
	unsigned mask = 0xFFFFFFFFU;
	int lane = threadIdx.x % warpSize;
	int warpID = threadIdx.x / warpSize;
	while (idx < order*order*order)
	{ // grid stride loop to load
		val += gdata[idx];
		idx += gridDim.x * blockDim.x;
	}
	// 1st warp-shuffle reduction
	for (int offset = warpSize / 2; offset > 0; offset >>= 1)
		val += __shfl_down_sync(mask, val, offset);
	if (lane == 0) sdata[warpID] = val;
	__syncthreads(); // put warp results in shared mem
	// hereafter, just warp 0
	if (warpID == 0)
	{
		// reload val from shared mem if warp existed
		val = (tid < blockDim.x / warpSize) ? sdata[lane] : 0;
		// final warp-shuffle reduction
		for (int offset = warpSize / 2; offset > 0; offset >>= 1)
			val += __shfl_down_sync(mask, val, offset);
		if (tid == 0) atomicAdd(out, val);
	}
}

bool fit()
{
	static int counter = 0;

	if (is_fit)
		return true;

	const static dim3 block(8, 8, 8);
	const static dim3 grid_r((res + block.x - 1) / block.x, (res + block.y - 1) / block.y, (res + block.z - 1) / block.z);
	const static dim3 grid_o((order + block.x - 1) / block.x, (order + block.y - 1) / block.y, (order + block.z - 1) / block.z);

	cuda_compute_node_velocity << <grid_r, block >> > (
		d_legendre,
		d_a,
		d_b,
		d_c,
		d_node_velocity_x,
		d_node_velocity_y,
		d_node_velocity_z
		);
//	cudaError("cuda_compute_node_velocity");

	cuda_fit << <grid_o, block >> > (
		d_legendre,
		d_velocity_x,
		d_velocity_y,
		d_velocity_z,
		d_node_velocity_x,
		d_node_velocity_y,
		d_node_velocity_z,
		d_a,
		d_b,
		d_c,
		d_velocity_mse
		);
//	cudaError("cuda_fit");


	if (counter % 10 == 0)
	{
		printf("fitting ... %d \t", counter);
		cudaMemcpyAsync(node_velocity_x.data(), d_node_velocity_x, sizeof(float) * res * res * res, cudaMemcpyDeviceToHost);
		cudaMemcpyAsync(node_velocity_y.data(), d_node_velocity_y, sizeof(float) * res * res * res, cudaMemcpyDeviceToHost);
		cudaMemcpyAsync(node_velocity_z.data(), d_node_velocity_z, sizeof(float) * res * res * res, cudaMemcpyDeviceToHost);
		cudaMemsetAsync(d_velocity_mse_total, 0, sizeof(float) * 1);
		reduce_ws << <(order*order*order + 1023)/1024, 1024 >> > (d_velocity_mse, d_velocity_mse_total);

		static float mse_total;
		cudaMemcpyAsync(&mse_total, d_velocity_mse_total, sizeof(float), cudaMemcpyDeviceToHost);
		printf("mse_total : %.3f \n", mse_total / (order * order * order));
		if (mse_total < 100.0f * order * order * order)
		{
			puts("fit!!");
			is_fit = true;
		}
	}

//	cudaError("cudaMemcpyAsync");

	counter++;
	return is_fit;
}


__global__ void cuda_step(
	const float* __restrict__ d_elements,
	const float* __restrict__ d_Ppq,
	const float* __restrict__ d_a,
	const float* __restrict__ d_b,
	const float* __restrict__ d_c,
	float* __restrict__ d_a_out,
	float* __restrict__ d_b_out,
	float* __restrict__ d_c_out
)
{
	volatile int o = threadIdx.x + blockIdx.x * blockDim.x;
	volatile int p = threadIdx.y + blockIdx.y * blockDim.y;
	volatile int q = threadIdx.z + blockIdx.z * blockDim.z;
	if (o >= order || p >= order || q >= order)
		return;
	
	float x_advect = 0.0f;
	float y_advect = 0.0f;
	float z_advect = 0.0f;
	float &xx = x_advect, &xy = x_advect, &xz = x_advect;
	float &yx = y_advect, &yy = y_advect, &yz = y_advect;
	float &zx = z_advect, &zy = z_advect, &zz = z_advect;
	for (int a = 0; a < order; a++)
		for (int b = 0; b < order; b++)
		{
			float e_ao = d_elements[IDX_o(a, b, o)];
		//	float e_bo = d_elements[IDX_o(b, a, o)];
			float Ppq_abo = d_Ppq[IDX_o(a, b, o)];

			if (e_ao == 0.0f && Ppq_abo == 0.0f)
				continue;

			for (int c = 0; c < order; c++)
				for (int d = 0; d < order; d++)
				{
					float e_cp = d_elements[IDX_o(c, d, p)];
				//	float e_dp = d_elements[IDX_o(d, c, p)];
					float Ppq_cdp = d_Ppq[IDX_o(c, d, p)];

					if (e_cp == 0.0f && Ppq_cdp == 0.0f)
						continue;

					for (int e = 0; e < order; e++)
					{
						int ace_o = IDX_o(a, c, e);
						float a_ace = d_a[ace_o];
						float b_ace = d_b[ace_o];
						float c_ace = d_c[ace_o];
						for (int f = 0; f < order; f++)
						{
							int bdf_o = IDX_o(b, d, f);
							float a_bdf = d_a[bdf_o];
							float b_bdf = d_b[bdf_o];
							float c_bdf = d_c[bdf_o];

							float e_eq = d_elements[IDX_o(e, f, q)];
						//	float e_fq = d_elements[IDX_o(f, e, q)];
							float Ppq_efq = d_Ppq[IDX_o(e, f, q)];

							if (e_eq == 0.0f && Ppq_efq == 0.0f)
								continue;

							float t;

							t = (e_ao) * Ppq_cdp * Ppq_efq * a_bdf;
							if (t != 0.0f)
							{
								xx += a_ace * t;
								yx += b_ace * t;
								zx += c_ace * t;
							}
							t = (e_cp) * Ppq_abo * Ppq_efq * b_bdf;
							if (t != 0.0f)
							{
								xy += a_ace * t;
								yy += b_ace * t;
								zy += c_ace * t;
							}
							t = (e_eq) * Ppq_abo * Ppq_cdp * c_bdf;
							if (t != 0.0f)
							{
								xz += a_ace * t;
								yz += b_ace * t;
								zz += c_ace * t;
							}
							//xx += e_ao * Ppq_cdp * Ppq_efq * a_ace * a_bdf;
							//xy += Ppq_abo * e_cp * Ppq_efq * a_ace * b_bdf;
							//xz += Ppq_abo * Ppq_cdp * e_eq * a_ace * c_bdf;
							//
							//yx += e_ao * Ppq_cdp * Ppq_efq * b_ace * a_bdf;
							//yy += Ppq_abo * e_cp * Ppq_efq * b_ace * b_bdf;
							//yz += Ppq_abo * Ppq_cdp * e_eq * b_ace * c_bdf;
							//
							//zx += e_ao * Ppq_cdp * Ppq_efq * c_ace * a_bdf;
							//zy += Ppq_abo * e_cp * Ppq_efq * c_ace * b_bdf;
							//zz += Ppq_abo * Ppq_cdp * e_eq * c_ace * c_bdf;
						}
					}
				}
		}
	int opq_o = IDX_o(o, p, q);
	d_a_out[opq_o] = d_a[opq_o] - (x_advect) * sim_step_size;
	d_b_out[opq_o] = d_b[opq_o] - (y_advect) * sim_step_size;
	d_c_out[opq_o] = d_c[opq_o] - (z_advect) * sim_step_size;
}

void step()
{
	static int counter = 0;


	const static dim3 block(16, 8, 8);
	const static dim3 grid_r((res + block.x - 1) / block.x, (res + block.y - 1) / block.y, (res + block.z - 1) / block.z);
	const static dim3 grid_o((order + block.x - 1) / block.x, (order + block.y - 1) / block.y, (order + block.z - 1) / block.z);

	cuda_step << <grid_o, block >> > (
		d_elements,
		d_Ppq,
		d_a,
		d_b,
		d_c,
		d_a2,
		d_b2,
		d_c2
		);
	std::swap(d_a, d_a2);
	std::swap(d_b, d_b2);
	std::swap(d_c, d_c2);

//	cudaError("cuda_step");

	if (counter % 10 == 0)
	{
		printf("step! ... %d \t", counter);
	//	static float *a = new float[order * order * order];
	//	static float *b = new float[order * order * order];
	//	static float *c = new float[order * order * order];
	//	cudaMemcpyAsync(a, d_a, sizeof(float) * order * order * order, cudaMemcpyDeviceToHost);
	//	cudaMemcpyAsync(b, d_b, sizeof(float) * order * order * order, cudaMemcpyDeviceToHost);
	//	cudaMemcpy(c, d_c, sizeof(float) * order * order * order, cudaMemcpyDeviceToHost);
	//	system("cls");
	//	for(int i = 0; i < order; i++)
	//		for (int j = 0; j < order; j++)
	//			for (int k = 0; k < order; k++)
	//			{
	//				float aa = a[IDX_o(i, j, k)];
	//				if(!(abs(aa) < 1e-3f))
	//					printf("%.5e ", aa);
	//				float bb = b[IDX_o(i, j, k)];
	//				if (!(abs(bb) < 1e-3f))
	//					printf("%.5e ", bb);
	//				float cc = c[IDX_o(i, j, k)];
	//				if (!(abs(cc) < 1e-3f))
	//					printf("%.5e ", cc);
	//			}

		cuda_compute_node_velocity << <grid_r, block >> > (
			d_legendre,
			d_a,
			d_b,
			d_c,
			d_node_velocity_x,
			d_node_velocity_y,
			d_node_velocity_z
			);
	//	cudaError("cuda_compute_node_velocity");

		cudaMemcpyAsync(node_velocity_x.data(), d_node_velocity_x, sizeof(float) * res * res * res, cudaMemcpyDeviceToHost);
		cudaMemcpyAsync(node_velocity_y.data(), d_node_velocity_y, sizeof(float) * res * res * res, cudaMemcpyDeviceToHost);
		cudaMemcpyAsync(node_velocity_z.data(), d_node_velocity_z, sizeof(float) * res * res * res, cudaMemcpyDeviceToHost);
	}

	
//	float a_step[order * order] = { 0, };
//	float b_step[order * order] = { 0, };
//
//
//
//#pragma omp parallel for
//	for (int op = 0; op < order * order; op++)
//	{
//		int o = op % order;
//		int p = op / order;
//
//		float xx = 0;
//		float xy = 0;
//		float yy = 0;
//		float yx = 0;
//
//		for (int a = 0; a < order; a++)
//		{
//			for (int b = 0; b < order; b++)
//			{
//				for (int c = 0; c < order; c++)
//				{
//					float a_ac = coeff.a[a + c * order];
//					float b_ac = coeff.b[a + c * order];
//
//					for (int d = 0; d < order; d++)
//					{
//						float a_bd = coeff.a[b + d * order];
//						float b_bd = coeff.b[b + d * order];
//
//						xx += elements[a][b][o] * Ppq[c][d][p] * a_ac * a_bd;
//						xy += elements[b][a][o] * Ppq[c][d][p] * a_ac * b_bd;
//						yy += elements[c][d][p] * Ppq[a][b][o] * b_ac * b_bd;
//						yx += elements[d][c][p] * Ppq[a][b][o] * b_ac * a_bd;
//
//					}
//				}
//
//			}
//		}
//
//		a_step[op] = xx + xy;
//		b_step[op] = yy + yx;
//	}
//
//	for (int op = 0; op < order * order; op++)
//	{
//		if (op == 0 && counter % 10 == 0)
//			printf("%.5lf %.5lf \n", coeff.a[op], coeff.b[op]);
//		coeff.a[op] -= a_step[op] * sim_step_size;
//		coeff.b[op] -= b_step[op] * sim_step_size;
//
//		//	coeff.a[op] *= 0.999;
//		//	coeff.b[op] *= 0.999;
//	}
//
//	for (int op = 0; op < order * order; op++)
//	{
//		int o = op % order;
//		int p = op / order;
//
//		float val_a = 0.0;
//		float val_b = 0.0;
//		for (int o2 = o + 1; o2 < order; o2 += 2)
//			val_a += coeff.a[o2 + p * order];
//		for (int p2 = p + 1; p2 < order; p2 += 2)
//			val_b += coeff.b[o + p2 * order];
//
//		a_step[op] = val_a / (2.0f * o + 1.0f);
//		b_step[op] = val_b / (2.0f * p + 1.0f);
//	}
//
//	for (int op = 0; op < order * order; op++)
//	{
//		coeff.a[op] += (-b_step[op] - coeff.a[op]) * sim_step_size;
//		coeff.b[op] += (-a_step[op] - coeff.b[op]) * sim_step_size;
//	}
//
//
	counter++;
}