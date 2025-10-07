//#include <cutil_inline.h>
#include <cstdlib>
#include <cstdio>
#include <string.h>
//#include <GL/freeglut.h>
#include <helper_cuda.h>
//#include <cuda_gl_interop.h>

#include <math.h>

__constant__ float pT, a, b, cc, alphas, Tc;

__device__ float4
cuda_computePhotonsQGPBI(float4 vT, float4 lambda) 
{
	extern __shared__ float4 phieta[];
	
	float4 ret = {0.0f, 0.0f, 0.0f, 0.0f};
	for(int iphi = 0; iphi < 32; ++iphi) {
		float sumphi = 0.f;
		for(int ieta = 0; ieta < 28; ++ieta) {
			float etil = (pT * phieta[32 + ieta].y - vT.x * pT * phieta[iphi].y - vT.y * pT * phieta[iphi].z) / sqrt(1. - vT.x*vT.x - vT.y*vT.y);
			if (etil<1.e10f) 
				sumphi += phieta[32 + ieta].w * exp(-etil / vT.w) * ( lambda.x*lambda.x*(log(a*etil/alphas/vT.w) + b*etil/vT.w) + lambda.x*log(cc*etil/alphas/vT.w) );
			//sumphi += phieta[32 + ieta].w * exp(-etil / vT.w) * lambda.x * lambda.x * log(a*etil/alphas/vT.w);
			//sumphi += phieta[32 + ieta].w * exp(-etil / vT.w);
			//printf("%f %f %f %f\n", etil, vT.x, vT.y, phieta[32 + ieta].y);
		}
		ret.x += phieta[iphi].w                          * sumphi;
		ret.y += phieta[iphi].w * cos(phieta[iphi].x)    * sumphi;
		ret.z += phieta[iphi].w * cos(2.*phieta[iphi].x) * sumphi;
		ret.w += phieta[iphi].w * cos(3.*phieta[iphi].x) * sumphi;
	}
	return ret;
}

__global__ void
cuda_PhotonsQGPBI(float4* Phi, float4* Eta, float4* vTs, float4* lambdas, float4* Res, int N, int calcmaxnumblast)
{
	extern __shared__ float4 phieta[];

	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index>=N)	index = N - 1;
	//Res[index] = {0.0f, 0.0f, 0.0f, 0.0f};

	if (threadIdx.x < 32) phieta[threadIdx.x]      = Phi[threadIdx.x];
	if (threadIdx.x < 32) phieta[32 + threadIdx.x] = Eta[threadIdx.x];
	__syncthreads();

	Res[index] = cuda_computePhotonsQGPBI(vTs[index], lambdas[index]);
}

__device__ float4
cuda_computePhotonsQGPBISymm(float4 vT, float4 lambda) 
{
	extern __shared__ float4 phieta[];
	
	float4 ret = {0.0f, 0.0f, 0.0f, 0.0f};
	//for(int iphi = 0; iphi < 32; ++iphi) {
		float sumphi = 0.f;
		for(int ieta = 0; ieta < 32; ++ieta) {
			float etil = (pT * phieta[32 + ieta].y - vT.x * pT) / sqrt(1. - vT.x*vT.x - vT.y*vT.y);
			sumphi += phieta[32 + ieta].w * exp(-etil / vT.w) * ( lambda.x*lambda.x*(log(a*etil/alphas/vT.w) + b*etil/vT.w) + lambda.x*log(cc*etil/alphas/vT.w) );
		}
		ret.x += 2.f * 3.1415926535f * sumphi;
		ret.y += 0.f;
		ret.z += 0.f;
		ret.w += 0.f;
	//}
	return ret;
}

__global__ void
cuda_PhotonsQGPBISymm(float4* Eta, float4* vTs, float4* lambdas, float4* Res, int N, int calcmaxnumblast)
{
	extern __shared__ float4 phieta[];

	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index>=N)	index = N - 1;
	//Res[index] = {0.0f, 0.0f, 0.0f, 0.0f};

	//if (threadIdx.x < 32) phieta[threadIdx.x]      = Phi[threadIdx.x];
	if (threadIdx.x < 32) phieta[32 + threadIdx.x] = Eta[threadIdx.x];
	__syncthreads();

	//Res[index] = 0.f;

	Res[index] = cuda_computePhotonsQGPBISymm(vTs[index], lambdas[index]);
}

__constant__ float A1, A2, B1, B2, Fq, Nf, aEM;

__device__ float
cuda_AMY_G1(float E, float T, float as)
{
	return A1 * Fq * aEM * as * T * T * exp(-E) * log(B1*E/as) * 659.55874088f;
}

__device__ float
cuda_AMY_G2(float E, float T, float as)
{
	return A2 * Fq * aEM * as * T * T * exp(-E) * log(B2*E/as) * 659.55874088f;
}


__device__ float
cuda_AMY_C12(float E)
{
	return (0.041f / E) - 0.3615f + 1.01f * exp(-1.35f * E);
}

__device__ float
cuda_AMY_C34(float E)
{
	return sqrt(1.f + Nf/6.f) * (0.548f / pow(E, 3.f/2.f) * log(12.28f + 1.f/E) + 0.133f * E / sqrt(1.f + E/16.27f));
}

__device__ float
cuda_AMY_G(float E, float T, float as)
{
	return 1.f / 3.1415926535f / 3.1415926535f * Fq * aEM * as * T * T / (exp(E) + 1) * (0.5f * log(3.*E/2./3.1415926535f/as) + cuda_AMY_C12(E) + cuda_AMY_C34(E)) * 659.55874088f;
}

__device__ float4
cuda_computePhotonsAMYBILow(float4 vT, float4 lambda) 
{
	extern __shared__ float4 phieta[];
	
	float4 ret = {0.0f, 0.0f, 0.0f, 0.0f};

	for(int iphi = 0; iphi < 32; ++iphi) {
		float sumphi = 0.f;
		for(int ieta = 0; ieta < 28; ++ieta) {
			float etil = (pT * phieta[32 + ieta].y - vT.x * pT * phieta[iphi].y - vT.y * pT * phieta[iphi].z) / sqrt(1. - vT.x*vT.x - vT.y*vT.y) / vT.w;
			if (etil<1.e10f) 
				sumphi += phieta[32 + ieta].w * (lambda.x * cuda_AMY_G1(etil, vT.w, lambda.y) + lambda.x * lambda.x * (cuda_AMY_G(etil, vT.w, lambda.y) - cuda_AMY_G1(etil, vT.w, lambda.y)) );
				//sumphi += phieta[32 + ieta].w * GL(etil/t, t, lambda, fMode);
			//sumphi += phieta[32 + ieta].w * exp(-etil / vT.w) * lambda.x * lambda.x * log(a*etil/alphas/vT.w);
			//sumphi += phieta[32 + ieta].w * exp(-etil / vT.w);
			//printf("%f %f %f %f\n", etil, vT.x, vT.y, phieta[32 + ieta].y);
		}
		ret.x += phieta[iphi].w                          * sumphi;
		ret.y += phieta[iphi].w * cos(phieta[iphi].x)    * sumphi;
		ret.z += phieta[iphi].w * cos(2.*phieta[iphi].x) * sumphi;
		ret.w += phieta[iphi].w * cos(3.*phieta[iphi].x) * sumphi;
	}
	return ret;
}

__device__ float4
cuda_computePhotonsAMYBIHigh(float4 vT, float4 lambda) 
{
	extern __shared__ float4 phieta[];
	
	float4 ret = {0.0f, 0.0f, 0.0f, 0.0f};

	for(int iphi = 0; iphi < 32; ++iphi) {
		float sumphi = 0.f;
		for(int ieta = 0; ieta < 28; ++ieta) {
			float etil = (pT * phieta[32 + ieta].y - vT.x * pT * phieta[iphi].y - vT.y * pT * phieta[iphi].z) / sqrt(1. - vT.x*vT.x - vT.y*vT.y) / vT.w;
			if (etil<1.e10f) 
				sumphi += phieta[32 + ieta].w * (lambda.x * lambda.x * cuda_AMY_G2(etil, vT.w, lambda.y) + lambda.x * (cuda_AMY_G(etil, vT.w, lambda.y) - cuda_AMY_G2(etil, vT.w, lambda.y)) );
				//sumphi += phieta[32 + ieta].w * GL(etil/t, t, lambda, fMode);
			//sumphi += phieta[32 + ieta].w * exp(-etil / vT.w) * lambda.x * lambda.x * log(a*etil/alphas/vT.w);
			//sumphi += phieta[32 + ieta].w * exp(-etil / vT.w);
			//printf("%f %f %f %f\n", etil, vT.x, vT.y, phieta[32 + ieta].y);
		}
		ret.x += phieta[iphi].w                          * sumphi;
		ret.y += phieta[iphi].w * cos(phieta[iphi].x)    * sumphi;
		ret.z += phieta[iphi].w * cos(2.*phieta[iphi].x) * sumphi;
		ret.w += phieta[iphi].w * cos(3.*phieta[iphi].x) * sumphi;
	}
	return ret;
}


__device__ float4
cuda_computePhotonsAMYBISymmLow(float4 vT, float4 lambda) 
{
	extern __shared__ float4 phieta[];
	
	float4 ret = {0.0f, 0.0f, 0.0f, 0.0f};

	//for(int iphi = 0; iphi < 32; ++iphi) {
		float sumphi = 0.f;
		for(int ieta = 0; ieta < 28; ++ieta) {
			float etil = (pT * phieta[32 + ieta].y - vT.x * pT) / sqrt(1. - vT.x*vT.x - vT.y*vT.y) / vT.w;
			if (etil<1.e10f) 
				sumphi += phieta[32 + ieta].w * (lambda.x * cuda_AMY_G1(etil, vT.w, lambda.y) + lambda.x * lambda.x * (cuda_AMY_G(etil, vT.w, lambda.y) - cuda_AMY_G1(etil, vT.w, lambda.y)) );
				//sumphi += phieta[32 + ieta].w * GL(etil/t, t, lambda, fMode);
			//sumphi += phieta[32 + ieta].w * exp(-etil / vT.w) * lambda.x * lambda.x * log(a*etil/alphas/vT.w);
			//sumphi += phieta[32 + ieta].w * exp(-etil / vT.w);
			//printf("%f %f %f %f\n", etil, vT.x, vT.y, phieta[32 + ieta].y);
		}
		ret.x += 2.f * 3.1415926535f * sumphi;
		ret.y += 0.f;
		ret.z += 0.f;
		ret.w += 0.f;
	//}
	return ret;
}

__device__ float4
cuda_computePhotonsAMYBISymmHigh(float4 vT, float4 lambda) 
{
	extern __shared__ float4 phieta[];
	
	float4 ret = {0.0f, 0.0f, 0.0f, 0.0f};

	//for(int iphi = 0; iphi < 32; ++iphi) {
		float sumphi = 0.f;
		for(int ieta = 0; ieta < 28; ++ieta) {
			float etil = (pT * phieta[32 + ieta].y - vT.x * pT) / sqrt(1. - vT.x*vT.x - vT.y*vT.y) / vT.w;
			if (etil<1.e10f) 
				sumphi += phieta[32 + ieta].w * (lambda.x * cuda_AMY_G2(etil, vT.w, lambda.y) + lambda.x * lambda.x * (cuda_AMY_G(etil, vT.w, lambda.y) - cuda_AMY_G2(etil, vT.w, lambda.y)) );
				//sumphi += phieta[32 + ieta].w * GL(etil/t, t, lambda, fMode);
			//sumphi += phieta[32 + ieta].w * exp(-etil / vT.w) * lambda.x * lambda.x * log(a*etil/alphas/vT.w);
			//sumphi += phieta[32 + ieta].w * exp(-etil / vT.w);
			//printf("%f %f %f %f\n", etil, vT.x, vT.y, phieta[32 + ieta].y);
		}
		ret.x += 2.f * 3.1415926535f * sumphi;
		ret.y += 0.f;
		ret.z += 0.f;
		ret.w += 0.f;
	//}
	return ret;
}

__global__ void
cuda_PhotonsAMYBILow(float4* Phi, float4* Eta, float4* vTs, float4* lambdas, float4* Res, int N, int calcmaxnumblast)
{
	extern __shared__ float4 phieta[];

	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index>=N)	index = N - 1;
	//Res[index] = {0.0f, 0.0f, 0.0f, 0.0f};

	if (threadIdx.x < 32) phieta[threadIdx.x]      = Phi[threadIdx.x];
	if (threadIdx.x < 32) phieta[32 + threadIdx.x] = Eta[threadIdx.x];
	__syncthreads();

	Res[index] = cuda_computePhotonsAMYBILow(vTs[index], lambdas[index]);
}


__global__ void
cuda_PhotonsAMYBIHigh(float4* Phi, float4* Eta, float4* vTs, float4* lambdas, float4* Res, int N, int calcmaxnumblast)
{
	extern __shared__ float4 phieta[];

	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index>=N)	index = N - 1;
	//Res[index] = {0.0f, 0.0f, 0.0f, 0.0f};

	if (threadIdx.x < 32) phieta[threadIdx.x]      = Phi[threadIdx.x];
	if (threadIdx.x < 32) phieta[32 + threadIdx.x] = Eta[threadIdx.x];
	__syncthreads();

	Res[index] = cuda_computePhotonsAMYBIHigh(vTs[index], lambdas[index]);
}

__global__ void
cuda_PhotonsAMYBISymmLow(float4* Eta, float4* vTs, float4* lambdas, float4* Res, int N, int calcmaxnumblast)
{
	extern __shared__ float4 phieta[];

	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index>=N)	index = N - 1;
	//Res[index] = {0.0f, 0.0f, 0.0f, 0.0f};

	//if (threadIdx.x < 32) phieta[threadIdx.x]      = Phi[threadIdx.x];
	if (threadIdx.x < 32) phieta[32 + threadIdx.x] = Eta[threadIdx.x];
	__syncthreads();

	//Res[index] = 0.f;

	Res[index] = cuda_computePhotonsAMYBISymmLow(vTs[index], lambdas[index]);
}

__global__ void
cuda_PhotonsAMYBISymmHigh(float4* Eta, float4* vTs, float4* lambdas, float4* Res, int N, int calcmaxnumblast)
{
	extern __shared__ float4 phieta[];

	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index>=N)	index = N - 1;
	//Res[index] = {0.0f, 0.0f, 0.0f, 0.0f};

	//if (threadIdx.x < 32) phieta[threadIdx.x]      = Phi[threadIdx.x];
	if (threadIdx.x < 32) phieta[32 + threadIdx.x] = Eta[threadIdx.x];
	__syncthreads();

	//Res[index] = 0.f;

	Res[index] = cuda_computePhotonsAMYBISymmHigh(vTs[index], lambdas[index]);
}


// HADRONS
__device__ float
cuda_PPR_PKK(float E, float T)
{
	//return (0.041f / E) - 0.3615f + 1.01f * exp(-1.35f * E);
	if (E>20.f) return 0.f;
	float RPIK   = 1.f / pow(T,3.f) * exp(-(5.4018f*pow(T,-0.6864f)-1.51f)*pow(2.f*T*E, 0.07f) - 0.91f*E/T);
	float RPIKs  = pow(T,3.75f) * exp(-0.35f/pow(2.f*T*E,1.05f) + (2.3894f*pow(T,0.03435f) - 3.222f)*E/T);
	float RPIKsK = pow(T,3.70f) * exp(-(6.096f*pow(T,1.889f)+1.0299f)/pow(2.f*T*E,-1.613f*pow(T,2.162f)+0.975f) - 0.96f*E/T);
	return RPIK + RPIKs + RPIKsK;
}

__device__ float
cuda_PPR_rho(float E, float T)
{
	if (E>20.f) return 0.f;
	float aT = -31.21f + 353.61f*T - 1739.4f*T*T + 3105.f * T * T * T;
	float bT = -5.513f - 42.2f*T + 333.f*T*T - 570.f*T*T*T;
	float cT = -6.153f + 57.f*T - 134.61f*T*T + 8.31f*T*T*T;
	return exp(aT*E + bT + cT / (E + 0.2f));
}

__device__ float
cuda_PPR_brem(float E, float T)
{
	if (E>20.f) return 0.f;
	float abT = -16.28f + 62.45f*T - 93.4f*T*T - 7.5f*T*T*T;
	float bbT = -35.54f + 414.8f*T - 2054.f*T*T + 3718.8f*T*T*T;
	float gbT = 0.7364f - 10.72f*T + 56.32f*T*T - 103.5f*T*T*T;
	float dbT = -2.51f + 58.152f*T - 318.24f*T*T + 610.7f*T*T*T;
	return exp(abT + bbT*E + gbT*E*E + dbT/(E+0.2f));
}

__device__ float
cuda_PPR_thoma(float E, float T)
{
	if (E>20.f) return 0.f;
	return 4.8f * pow(T, 2.15f) * exp(-1.f/pow(1.35f*T*E,0.77f)) * exp(-E/T);
}


__device__ float
cuda_PPR_prgo(float E, float T)
{
	if (E>20.f) return 0.f;
	float a1 = -35.8991f + 460.425f * T - 2592.04f * T * T + 5342.32f * T * T * T;
	float a2 = -41.9725f + 601.952f * T - 3587.8f  * T * T + 7604.97f * T * T * T;
	float a3 = 0.740436f - 16.7159f * T + 133.526f * T * T - 347.589f * T * T * T;
	float a4 = 2.00611f - 3.79343f * T + 29.3101f * T * T - 72.8725f * T * T * T;
	float a5 = -8.33046f + 121.091f * T - 801.676f * T * T + 1712.16f * T * T * T;
	float a6 = 17.9029f - 388.5f * T + 2779.03f * T * T - 6448.4f * T * T * T;
	float a7 = -15.622f + 340.651f * T - 2483.18f * T * T + 5870.61f * T * T * T;
	return exp(a1 * E + a2 + a3 * pow(E, a4) + a5 * pow(E + a6, a7));
}

__device__ float
cuda_PPR_pogr(float E, float T)
{
	if (E>20.f) return 0.f;
	float a1 = -29.4663f + 291.356f * T - 1301.27f * T * T + 2102.12f * T * T * T;
	float a2 = -45.081f + 688.929f * T - 4150.15f  * T * T +  8890.76f * T * T * T;
	float a3 = -0.260076f + 8.92875f * T - 60.868f * T * T +  136.57f * T * T * T;
	float a4 = 2.2663f - 8.30596f * T +  49.3342f * T * T - 90.8501f * T * T * T;
	float a5 = 10.2955f - 317.077f * T + 2412.15f * T * T - 6020.9f * T * T * T;
	float a6 =  3.12251f - 47.5277f * T + 222.61f * T * T - 241.9f * T * T * T;
	float a7 = -3.39045f + 56.5927f * T - 336.97f * T * T + 622.756f * T * T * T;
	return exp(a1 * E + a2 + a3 * pow(E, a4) + a5 * pow(E + a6, a7));
}

__device__ float
cuda_PPR_rogp(float E, float T)
{
	if (E>20.f) return 0.f;
	float a1 = -29.6866f + 331.769f * T - 1618.66f * T * T + 2918.53f * T * T * T;
	float a2 = -15.3332f + 90.2225f * T - 300.185f  * T * T + 428.386f * T * T * T;
	float a3 = -7.35061f + 109.288f * T - 630.396f * T * T + 1227.69f * T * T * T;
	float a4 = -10.6044f + 109.1f * T - 500.718f * T * T + 872.951f * T * T * T;
	return exp(a1 * E + a2 + a3 / (E + 0.2f) + a4  / (E + 0.2f) / (E + 0.2f));
}

__device__ float
cuda_PPR_pirhoomega(float E, float T)
{
	return cuda_PPR_prgo(E,T) + cuda_PPR_pogr(E,T) + cuda_PPR_rogp(E,T);
}

__device__ float
cuda_PPR(float E, float T)
{
	//return cuda_PPR_PKK(E,T) + cuda_PPR_rho(E,T) + cuda_PPR_brem(E,T) + cuda_PPR_thoma(E,T); 
	//return cuda_PPR_PKK(E,T) + cuda_PPR_rho(E,T) + cuda_PPR_brem(E,T); 
	return cuda_PPR_PKK(E,T) + cuda_PPR_rho(E,T) + cuda_PPR_brem(E,T) + cuda_PPR_pirhoomega(E,T); 
	//return cuda_PPR_PKK(E,T); 
	//return cuda_PPR_rho(E,T); 
	//if (T<0.155f && cuda_PPR_brem(E,T)>1.) printf("%lf %lf %lf\n", E, T, cuda_PPR_brem(E,T));
	//return cuda_PPR_brem(E,T); 
}

__device__ float4
cuda_computePhotonsHadronsBI(float4 vT, float4 lambda) 
{
	extern __shared__ float4 phieta[];
	
	float4 ret = {0.0f, 0.0f, 0.0f, 0.0f};

	for(int iphi = 0; iphi < 32; ++iphi) {
		float sumphi = 0.f;
		for(int ieta = 0; ieta < 28; ++ieta) {
			float etil = (pT * phieta[32 + ieta].y - vT.x * pT * phieta[iphi].y - vT.y * pT * phieta[iphi].z) / sqrt(1. - vT.x*vT.x - vT.y*vT.y);// / vT.w;
			if (etil<1.e10f) 
				sumphi += phieta[32 + ieta].w * cuda_PPR(etil, vT.w);
				//sumphi += phieta[32 + ieta].w * GL(etil/t, t, lambda, fMode);
			//sumphi += phieta[32 + ieta].w * exp(-etil / vT.w) * lambda.x * lambda.x * log(a*etil/alphas/vT.w);
			//sumphi += phieta[32 + ieta].w * exp(-etil / vT.w);
			//printf("%f %f %f %f\n", etil, vT.x, vT.y, phieta[32 + ieta].y);
		}
		ret.x += phieta[iphi].w                          * sumphi;
		ret.y += phieta[iphi].w * cos(phieta[iphi].x)    * sumphi;
		ret.z += phieta[iphi].w * cos(2.*phieta[iphi].x) * sumphi;
		ret.w += phieta[iphi].w * cos(3.*phieta[iphi].x) * sumphi;
	}
	return ret;
}

__device__ float4
cuda_computePhotonsHadronsBISymm(float4 vT, float4 lambda) 
{
	extern __shared__ float4 phieta[];
	
	float4 ret = {0.0f, 0.0f, 0.0f, 0.0f};

	//for(int iphi = 0; iphi < 32; ++iphi) {
		float sumphi = 0.f;
		for(int ieta = 0; ieta < 28; ++ieta) {
			float etil = (pT * phieta[32 + ieta].y - vT.x * pT) / sqrt(1. - vT.x*vT.x - vT.y*vT.y);// / vT.w;
			if (etil<1.e10f) 
				sumphi += phieta[32 + ieta].w * cuda_PPR(etil, vT.w);
				//sumphi += phieta[32 + ieta].w * GL(etil/t, t, lambda, fMode);
			//sumphi += phieta[32 + ieta].w * exp(-etil / vT.w) * lambda.x * lambda.x * log(a*etil/alphas/vT.w);
			//sumphi += phieta[32 + ieta].w * exp(-etil / vT.w);
			//printf("%f %f %f %f\n", etil, vT.x, vT.y, phieta[32 + ieta].y);
		}
		ret.x += 2.f * 3.1415926535f * sumphi;
		ret.y += 0.f;
		ret.z += 0.f;
		ret.w += 0.f;
	//}
	return ret;
}

__global__ void
cuda_PhotonsHadronsBI(float4* Phi, float4* Eta, float4* vTs, float4* lambdas, float4* Res, int N, int calcmaxnumblast)
{
	extern __shared__ float4 phieta[];

	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index>=N)	index = N - 1;
	//Res[index] = {0.0f, 0.0f, 0.0f, 0.0f};

	if (threadIdx.x < 32) phieta[threadIdx.x]      = Phi[threadIdx.x];
	if (threadIdx.x < 32) phieta[32 + threadIdx.x] = Eta[threadIdx.x];
	__syncthreads();

	Res[index] = cuda_computePhotonsHadronsBI(vTs[index], lambdas[index]);
}

__global__ void
cuda_PhotonsHadronsBISymm(float4* Eta, float4* vTs, float4* lambdas, float4* Res, int N, int calcmaxnumblast)
{
	extern __shared__ float4 phieta[];

	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index>=N)	index = N - 1;
	//Res[index] = {0.0f, 0.0f, 0.0f, 0.0f};

	//if (threadIdx.x < 32) phieta[threadIdx.x]      = Phi[threadIdx.x];
	if (threadIdx.x < 32) phieta[32 + threadIdx.x] = Eta[threadIdx.x];
	__syncthreads();

	Res[index] = cuda_computePhotonsHadronsBISymm(vTs[index], lambdas[index]);
}

__global__ void
cuda_PhotonsAMYHighVsHadronsBI(float4* Phi, float4* Eta, float4* vTs, float4* lambdas, float4* Res, int N, int calcmaxnumblast)
{
	extern __shared__ float4 phieta[];

	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index>=N)	index = N - 1;
	//Res[index] = {0.0f, 0.0f, 0.0f, 0.0f};

	if (threadIdx.x < 32) phieta[threadIdx.x]      = Phi[threadIdx.x];
	if (threadIdx.x < 32) phieta[32 + threadIdx.x] = Eta[threadIdx.x];
	__syncthreads();

	float4 ret1 = cuda_computePhotonsAMYBIHigh(vTs[index], lambdas[index]);
	float4 ret2 = cuda_computePhotonsHadronsBI(vTs[index], lambdas[index]);
	if (vTs[index].w > Tc) Res[index] = ret1;
	else Res[index] = ret2;
}

extern "C"
{

void cuda_allocateArray(float** dest, int number)
{
	// 4 floats each for alignment reasons
	unsigned int memSize = sizeof(float) * 4 * number;
    
    checkCudaErrors(cudaMalloc((void**)dest, memSize));
}


void cuda_deleteArray(float* arr)
{
    checkCudaErrors(cudaFree((void**)arr));
}

void cuda_copyArrayFromDevice(float* host, 
                         const float* device, 
                         unsigned int pbo, 
                         int numBodies)
{   
    //if (pbo)
    //    checkCudaErrors(cudaGLMapBufferObject((void**)&device, pbo));
    checkCudaErrors(cudaMemcpy(host, device, numBodies*4*sizeof(float),
                              cudaMemcpyDeviceToHost));
    //if (pbo)
    //   checkCudaErrors(cudaGLUnmapBufferObject(pbo));
}

void cuda_copyArrayToDevice(float* device, const float* host, int numBodies)
{
    checkCudaErrors(cudaMemcpy(device, host, numBodies*4*sizeof(float),
                              cudaMemcpyHostToDevice));
}

void cuda_threadSync() { checkCudaErrors(cudaDeviceSynchronize()); }

void 
cuda_addCellsBI_PhotonsQGP(float* Phi, float* Eta, float* vTs, float* lambdas, float* Res,
                     int N, float pT_, float a_, float b_, float cc_, float alphas_, int p, int q)
{
	int sharedMemSize = 64 * sizeof(float4);
	int grids = (int)((N+p-1)/p);
	dim3 dimGrid(grids);
	dim3 dimBlock(p);
	int calcmaxnumblast = N % p;
	calcmaxnumblast = (calcmaxnumblast==0) ? p : calcmaxnumblast;

    checkCudaErrors(cudaMemcpyToSymbol(pT,
                                     &pT_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(a,
                                     &a_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(b,
                                     &b_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(cc,
                                     &cc_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(alphas,
                                     &alphas_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));


	float time, cumulative_time = 0.f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cuda_PhotonsQGPBI<<< dimGrid, dimBlock, sharedMemSize >>>
            ((float4*)Phi, (float4*)Eta, (float4*)vTs, (float4*)lambdas, (float4*)Res,
            N, calcmaxnumblast);

	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

	printf("Kernel time = %lf ms\n", time);	
    
    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution failed");
}

void 
cuda_addCellsBISymm_PhotonsQGP(float* Eta, float* vTs, float* lambdas, float* Res,
                     int N, float pT_, float a_, float b_, float cc_, float alphas_, int p, int q)
{
	int sharedMemSize = 64 * sizeof(float4);
	int grids = (int)((N+p-1)/p);
	dim3 dimGrid(grids);
	dim3 dimBlock(p);
	int calcmaxnumblast = N % p;
	calcmaxnumblast = (calcmaxnumblast==0) ? p : calcmaxnumblast;

    checkCudaErrors(cudaMemcpyToSymbol(pT,
                                     &pT_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(a,
                                     &a_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(b,
                                     &b_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(cc,
                                     &cc_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(alphas,
                                     &alphas_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));


	/*float time, cumulative_time = 0.f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord(start, 0);*/

	cuda_PhotonsQGPBISymm<<< dimGrid, dimBlock, sharedMemSize >>>
            ((float4*)Eta, (float4*)vTs, (float4*)lambdas, (float4*)Res,
            N, calcmaxnumblast);

	/*cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

	printf("Kernel time = %lf ms\n", time);*/
    
    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution failed");
}


void 
cuda_addCellsBI_PhotonsAMYLow(float* Phi, float* Eta, float* vTs, float* lambdas, float* Res,
                     int N, float pT_, float A1_, float A2_, float B1_, float B2_, float Fq_, float Nf_, float aEM_, int p, int q)
{
	int sharedMemSize = 64 * sizeof(float4);
	int grids = (int)((N+p-1)/p);
	dim3 dimGrid(grids);
	dim3 dimBlock(p);
	int calcmaxnumblast = N % p;
	calcmaxnumblast = (calcmaxnumblast==0) ? p : calcmaxnumblast;

	checkCudaErrors(cudaMemcpyToSymbol(pT,
                                     &pT_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(A1,
                                     &A1_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(A2,
                                     &A2_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(B1,
                                     &B1_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(B2,
                                     &B2_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(Fq,
                                     &Fq_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(Nf,
                                     &Nf_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(aEM,
                                     &aEM_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));


	float time, cumulative_time = 0.f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cuda_PhotonsAMYBILow<<< dimGrid, dimBlock, sharedMemSize >>>
            ((float4*)Phi, (float4*)Eta, (float4*)vTs, (float4*)lambdas, (float4*)Res,
            N, calcmaxnumblast);

	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

	printf("Kernel time = %lf ms\n", time);	
    
    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution failed");
}


void 
cuda_addCellsBI_PhotonsAMYHigh(float* Phi, float* Eta, float* vTs, float* lambdas, float* Res,
                     int N, float pT_, float A1_, float A2_, float B1_, float B2_, float Fq_, float Nf_, float aEM_, int p, int q)
{
	int sharedMemSize = 64 * sizeof(float4);
	int grids = (int)((N+p-1)/p);
	dim3 dimGrid(grids);
	dim3 dimBlock(p);
	int calcmaxnumblast = N % p;
	calcmaxnumblast = (calcmaxnumblast==0) ? p : calcmaxnumblast;

	checkCudaErrors(cudaMemcpyToSymbol(pT,
                                     &pT_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(A1,
                                     &A1_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(A2,
                                     &A2_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(B1,
                                     &B1_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(B2,
                                     &B2_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(Fq,
                                     &Fq_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(Nf,
                                     &Nf_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(aEM,
                                     &aEM_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));


	float time, cumulative_time = 0.f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cuda_PhotonsAMYBIHigh<<< dimGrid, dimBlock, sharedMemSize >>>
            ((float4*)Phi, (float4*)Eta, (float4*)vTs, (float4*)lambdas, (float4*)Res,
            N, calcmaxnumblast);

	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

	printf("Kernel time = %lf ms\n", time);	
    
    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution failed");
}


void 
cuda_addCellsBISymm_PhotonsAMYLow(float* Eta, float* vTs, float* lambdas, float* Res,
                     int N, float pT_, float A1_, float A2_, float B1_, float B2_, float Fq_, float Nf_, float aEM_, int p, int q)
{
	int sharedMemSize = 64 * sizeof(float4);
	int grids = (int)((N+p-1)/p);
	dim3 dimGrid(grids);
	dim3 dimBlock(p);
	int calcmaxnumblast = N % p;
	calcmaxnumblast = (calcmaxnumblast==0) ? p : calcmaxnumblast;

	checkCudaErrors(cudaMemcpyToSymbol(pT,
                                     &pT_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(A1,
                                     &A1_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(A2,
                                     &A2_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(B1,
                                     &B1_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(B2,
                                     &B2_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(Fq,
                                     &Fq_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(Nf,
                                     &Nf_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(aEM,
                                     &aEM_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));


	float time, cumulative_time = 0.f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cuda_PhotonsAMYBISymmLow<<< dimGrid, dimBlock, sharedMemSize >>>
            ((float4*)Eta, (float4*)vTs, (float4*)lambdas, (float4*)Res,
            N, calcmaxnumblast);

	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

	printf("Kernel time = %lf ms\n", time);	
    
    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution failed");
}

void 
cuda_addCellsBISymm_PhotonsAMYHigh(float* Eta, float* vTs, float* lambdas, float* Res,
                     int N, float pT_, float A1_, float A2_, float B1_, float B2_, float Fq_, float Nf_, float aEM_, int p, int q)
{
	int sharedMemSize = 64 * sizeof(float4);
	int grids = (int)((N+p-1)/p);
	dim3 dimGrid(grids);
	dim3 dimBlock(p);
	int calcmaxnumblast = N % p;
	calcmaxnumblast = (calcmaxnumblast==0) ? p : calcmaxnumblast;

	checkCudaErrors(cudaMemcpyToSymbol(pT,
                                     &pT_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(A1,
                                     &A1_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(A2,
                                     &A2_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(B1,
                                     &B1_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(B2,
                                     &B2_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(Fq,
                                     &Fq_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(Nf,
                                     &Nf_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(aEM,
                                     &aEM_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));


	float time, cumulative_time = 0.f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cuda_PhotonsAMYBISymmHigh<<< dimGrid, dimBlock, sharedMemSize >>>
            ((float4*)Eta, (float4*)vTs, (float4*)lambdas, (float4*)Res,
            N, calcmaxnumblast);

	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

	printf("Kernel time = %lf ms\n", time);	
    
    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution failed");
}

void 
cuda_addCellsBI_PhotonsHadrons(float* Phi, float* Eta, float* vTs, float* lambdas, float* Res,
                     int N, float pT_, int p, int q)
{
	int sharedMemSize = 64 * sizeof(float4);
	int grids = (int)((N+p-1)/p);
	dim3 dimGrid(grids);
	dim3 dimBlock(p);
	int calcmaxnumblast = N % p;
	calcmaxnumblast = (calcmaxnumblast==0) ? p : calcmaxnumblast;

	checkCudaErrors(cudaMemcpyToSymbol(pT,
                                     &pT_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));


	float time, cumulative_time = 0.f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cuda_PhotonsHadronsBI<<< dimGrid, dimBlock, sharedMemSize >>>
            ((float4*)Phi, (float4*)Eta, (float4*)vTs, (float4*)lambdas, (float4*)Res,
            N, calcmaxnumblast);

	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

	printf("Kernel time = %lf ms\n", time);	
    
    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution failed");
}

void 
cuda_addCellsBISymm_PhotonsHadrons(float* Eta, float* vTs, float* lambdas, float* Res,
                     int N, float pT_, int p, int q)
{
	int sharedMemSize = 64 * sizeof(float4);
	int grids = (int)((N+p-1)/p);
	dim3 dimGrid(grids);
	dim3 dimBlock(p);
	int calcmaxnumblast = N % p;
	calcmaxnumblast = (calcmaxnumblast==0) ? p : calcmaxnumblast;

	checkCudaErrors(cudaMemcpyToSymbol(pT,
                                     &pT_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));


	float time, cumulative_time = 0.f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cuda_PhotonsHadronsBISymm<<< dimGrid, dimBlock, sharedMemSize >>>
            ((float4*)Eta, (float4*)vTs, (float4*)lambdas, (float4*)Res,
            N, calcmaxnumblast);

	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

	printf("Kernel time = %lf ms\n", time);	
    
    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution failed");
}

void 
cuda_addCellsBI_PhotonsMixAMYHigh(float* Phi, float* Eta, float* vTs, float* lambdas, float* Res,
                     int N, float pT_, float A1_, float A2_, float B1_, float B2_, float Fq_, float Nf_, float aEM_, float Tc_, int p, int q)
{
	int sharedMemSize = 64 * sizeof(float4);
	int grids = (int)((N+p-1)/p);
	dim3 dimGrid(grids);
	dim3 dimBlock(p);
	int calcmaxnumblast = N % p;
	calcmaxnumblast = (calcmaxnumblast==0) ? p : calcmaxnumblast;

	checkCudaErrors(cudaMemcpyToSymbol(pT,
                                     &pT_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(A1,
                                     &A1_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(A2,
                                     &A2_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(B1,
                                     &B1_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(B2,
                                     &B2_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(Fq,
                                     &Fq_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(Nf,
                                     &Nf_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(aEM,
                                     &aEM_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(Tc,
                                     &Tc_,
                                     sizeof(float), 0, 
                                     cudaMemcpyHostToDevice));


	float time, cumulative_time = 0.f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cuda_PhotonsAMYHighVsHadronsBI<<< dimGrid, dimBlock, sharedMemSize >>>
            ((float4*)Phi, (float4*)Eta, (float4*)vTs, (float4*)lambdas, (float4*)Res,
            N, calcmaxnumblast);

	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

	printf("Kernel time = %lf ms\n", time);	
    
    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution failed");
}

void cuda_deviceReset()
{
	cudaDeviceReset();
}

}
