#include "photonsCUDA.h"
#include "inc.h"
#include "NumericalIntegration.h"
#include "fld.h"
#include "eos.h"
#include "cll.h"

#ifdef USE_CUDA_TOOLKIT
extern "C"
{
    void cuda_allocateArray(float** dest, int number);
    void cuda_deleteArray(float* arr);
    void cuda_addCellsBI_PhotonsQGP(float* Phi, float* Eta, float* vTs, float* lambdas, float* Res,
                     int N, float pT, float a, float b, float cc, float alphas, int p, int q);
	void cuda_addCellsBISymm_PhotonsQGP(float* Eta, float* vTs, float* lambdas, float* Res,
                     int N, float pT, float a, float b, float cc, float alphas, int p, int q);
	void cuda_addCellsBI_PhotonsAMYHigh(float* Phi, float* Eta, float* vTs, float* lambdas, float* Res,
                     int N, float pT, float A1, float A2, float B1, float B2, float Fq, float Nf, float aEM, int p, int q);
	void cuda_addCellsBI_PhotonsAMYLow(float* Phi, float* Eta, float* vTs, float* lambdas, float* Res,
                     int N, float pT, float A1, float A2, float B1, float B2, float Fq, float Nf, float aEM, int p, int q);
	void cuda_addCellsBISymm_PhotonsAMYHigh(float* Eta, float* vTs, float* lambdas, float* Res,
                     int N, float pT, float A1, float A2, float B1, float B2, float Fq, float Nf, float aEM, int p, int q);
	void cuda_addCellsBISymm_PhotonsAMYLow(float* Eta, float* vTs, float* lambdas, float* Res,
                     int N, float pT, float A1, float A2, float B1, float B2, float Fq, float Nf, float aEM, int p, int q);
	void cuda_addCellsBI_PhotonsHadrons(float* Phi, float* Eta, float* vTs, float* lambdas, float* Res,
                     int N, float pT, int p, int q);
	void cuda_addCellsBISymm_PhotonsHadrons(float* Eta, float* vTs, float* lambdas, float* Res,
                     int N, float pT, int p, int q);
	void cuda_addCellsBI_PhotonsMixAMYHigh(float* Phi, float* Eta, float* vTs, float* lambdas, float* Res,
                     int N, float pT, float A1, float A2, float B1, float B2, float Fq, float Nf, float aEM, float Tc, int p, int q);
    void cuda_copyArrayFromDevice(float* host, const float* device, unsigned int pbo, int numBodies);
    void cuda_copyArrayToDevice(float* device, const float* host, int numBodies);
    void cuda_threadSync();
}
#else
	void cudaallocateArray(float** dest, int number) {}
	void cuda_deleteArray(float* arr) {}
	void cuda_addCellsBI_PhotonsQGP(float* Phi, float* Eta, float* vTs, float* lambdas, float* Res,
											int N, float pT, float a, float b, float cc, float alphas, int p, int q) {}
	void cuda_addCellsBISymm_PhotonsQGP(float* Eta, float* vTs, float* lambdas, float* Res,
											int N, float pT, float a, float b, float cc, float alphas, int p, int q) {}
	void cuda_addCellsBI_PhotonsAMYHigh(float* Phi, float* Eta, float* vTs, float* lambdas, float* Res,
											int N, float pT, float A1, float A2, float B1, float B2, float Fq, float Nf, float aEM, int p, int q) {}
	void cuda_addCellsBI_PhotonsAMYLow(float* Phi, float* Eta, float* vTs, float* lambdas, float* Res,
											int N, float pT, float A1, float A2, float B1, float B2, float Fq, float Nf, float aEM, int p, int q) {}
	void cuda_addCellsBISymm_PhotonsAMYHigh(float* Eta, float* vTs, float* lambdas, float* Res,
											int N, float pT, float A1, float A2, float B1, float B2, float Fq, float Nf, float aEM, int p, int q) {}
	void cuda_addCellsBISymm_PhotonsAMYLow(float* Eta, float* vTs, float* lambdas, float* Res,
											int N, float pT, float A1, float A2, float B1, float B2, float Fq, float Nf, float aEM, int p, int q) {}
	void cuda_addCellsBI_PhotonsHadrons(float* Phi, float* Eta, float* vTs, float* lambdas, float* Res,
											int N, float pT, int p, int q) {}
	void cuda_addCellsBISymm_PhotonsHadrons(float* Eta, float* vTs, float* lambdas, float* Res,
											int N, float pT, int p, int q) {}
	void cuda_addCellsBI_PhotonsMixAMYHigh(float* Phi, float* Eta, float* vTs, float* lambdas, float* Res,
											int N, float pT, float A1, float A2, float B1, float B2, float Fq, float Nf, float aEM, float Tc, int p, int q) {}
	void cuda_copyArrayFromDevice(float* host, const float* device, unsigned int pbo, int numBodies) {}
	void cuda_copyArrayToDevice(float* device, const float* host, int numBodies) {}
	void cuda_threadSync() {}
#endif


// QGP

PhotonsQGPCUDA::PhotonsQGPCUDA(const std::vector<double> & ptin, const std::vector<double> & yin, double tau0_, double taus_, double Tcut_) : Photons(ptin, yin, Tcut_), tau0(tau0_), taus(taus_)
{
	fN = 0;
	fCUDA = 0;
}

void PhotonsQGPCUDA::init() {
	coef = 7.297352e-3 * 0.3 / pow(C_PI, 3.) / 4. * (2. / 3.);
	coefBI = coef * 2.;
	//a = 0.197; b = 0.987; cc = 0.884;
	a = 0.232; b = 0.987; cc = 0.884;
	alphas = 0.3;

	GetCoefsIntegrateLegendre32(0., 2*C_PI, xphi, wphi);
	GetCoefsIntegrateLaguerre32(xeta, weta);
}

void PhotonsQGPCUDA::initCUDAArrays(Fluid *f) {
	fCUDA = 0;
#ifdef USE_CUDA_TOOLKIT
	if (f!=NULL) {
		fN = (f->getNX()-4) * (f->getNY()-4) * (f->getNZ()-4);
		h_vT     = new float[4 * fN];
		h_lambda = new float[4 * fN];
		h_res    = new float[4 * fN];
		h_phi    = new float[32 * 4];
		h_eta    = new float[32 * 4];

		for(int i = 0; i < 32; ++i) {
			double tphi = xphi[i];
			h_phi[4*i]     = (float)tphi;
			h_phi[4*i + 1] = (float)cos(tphi);
			h_phi[4*i + 2] = (float)sin(tphi);
			h_phi[4*i + 3] = (float)wphi[i];

			double teta = xeta[i];
			h_eta[4*i]     = (float)teta;
			h_eta[4*i + 1] = (float)cosh(teta);
			h_eta[4*i + 2] = (float)sinh(teta);
			h_eta[4*i + 3] = (float)weta[i];

			printf("%20E%20E%20E%20E\n", h_eta[4*i], h_eta[4*i+1], h_eta[4*i+2], h_eta[4*i+3]);
		}

		cuda_allocateArray(&d_vT, fN);
		cuda_allocateArray(&d_lambda, fN);
		cuda_allocateArray(&d_res, fN);
		cuda_allocateArray(&d_phi, 32);
		cuda_allocateArray(&d_eta, 32);

		cuda_copyArrayToDevice(d_phi, h_phi, 32);
		cuda_copyArrayToDevice(d_eta, h_eta, 32);
		
		fCUDA = 1;
	}
#endif
}

/*
void PhotonsQGPCUDA::addCells(double tau, double dtau, Fluid *f, EoS *eos, Cell *c) {
	double e, p, nb, nq, ns, t, mub, muq, mus, vx, vy, vz, s, Q[7], tauP;
	tauP = c->getTauP();
	f->getCMFvariables(c, tau, e, nb, nq, ns, vx, vy, vz);
	eos->eos(e, nb, nq, ns, t, mub, muq, mus, p, tauP);
	double lambda = 1.;
	if (taus>0.0) lambda = 1. - exp((tau0-tauP)/taus);
	//s = eos->s(e, nb, nq, ns, tauP);

	if (t<Tcut) return;

	// Sum over all pT and Y values
	for(int ic = 0; ic < pts.size(); ++ic) {
		double pt = pts[ic], Y = ys[ic];
		double sum = 0., sumv1 = 0., sumv2 = 0., sumv3 = 0.;
		double eta = c->getZ();
		double gamma = 1. / sqrt(1. - vx*vx - vy*vy - tanh(vz)*tanh(vz));
		double Etil1 = gamma * pt * (cosh(Y) - tanh(vz) * sinh(Y)); 
		// Integral over phi
		for(int iphi = 0; iphi < xphi.size(); ++iphi) {
			double tphi = xphi[iphi];
			double cosphi = cos(tphi), sinphi = sin(tphi);
			double sumphi = 0.;
			double Etil2 = -gamma * pt * (vx*cosphi + vy*sinphi);
			double etil = Etil1 + Etil2;
			sumphi += exp(-etil / t) * ( lambda*lambda*(log(a*etil/alphas/t) + b*etil/t) + lambda*log(cc*etil/alphas/t) );
			sum    += wphi[iphi]                * sumphi;
			sumv1  += wphi[iphi] * cos(tphi)    * sumphi;
			sumv2  += wphi[iphi] * cos(2.*tphi) * sumphi;
			sumv3  += wphi[iphi] * cos(3.*tphi) * sumphi;
		}

		double tcoef = coef * f->getDx() * f->getDy() * f->getDz() * tau * dtau * t * t * pow(gevtofm, 4);
		sum       *= tcoef;
		sumv1     *= tcoef;
		sumv2     *= tcoef;
		sumv3     *= tcoef;
		yield[ic] += sum;
		v1[ic]    += sumv1;
		v2[ic]    += sumv2;
		v3[ic]    += sumv3;
	}
}*/

PhotonsQGPCUDA::~PhotonsQGPCUDA() {
	if (fCUDA == 1) {
		delete [] h_vT;
		delete [] h_lambda;
		delete [] h_res;
		delete [] h_phi;
		delete [] h_eta;
		cuda_deleteArray(d_vT);
		cuda_deleteArray(d_lambda);
		cuda_deleteArray(d_res);
		cuda_deleteArray(d_phi);
		cuda_deleteArray(d_eta);
		//cuda_threadExit();
	}
}

void PhotonsQGPCUDA::addCellBI(double tau, double dtau, Fluid *f, EoS *eos, Cell *c) {
	double e, p, nb, nq, ns, t, mub, muq, mus, vx, vy, vz, s, Q[7], tauP;
	tauP = c->getTauP();
	f->getCMFvariables(c, tau, e, nb, nq, ns, vx, vy, vz);
	eos->eos(e, nb, nq, ns, t, mub, muq, mus, p, tauP);
	double lambda = 1.;
	if (taus>0.0) lambda = 1. - exp((tau0-tauP)/taus);
	//s = eos->s(e, nb, nq, ns, tauP);

	if (t<Tcut) return;

	// Sum over all pT values
	for(int ic = 0; ic < pts.size(); ++ic) {
		double pt = pts[ic];
		double sum = 0., sumv1 = 0., sumv2 = 0., sumv3 = 0.;
		// Integral over phi
		for(int iphi = 0; iphi < xphi.size(); ++iphi) {
			double tphi = xphi[iphi];
			double cosphi = cos(tphi), sinphi = sin(tphi);
			double sumphi = 0.;
			// Integral over eta
			for(int ieta = 0; ieta < xeta.size(); ++ieta) {
				double etil = (pt * cosh(xeta[ieta]) - vx * pt * cosphi - vy * pt * sinphi) / sqrt(1. - vx*vx - vy*vy);
				sumphi += weta[ieta] * exp(-etil / t) * ( lambda*lambda*(log(a*etil/alphas/t) + b*etil/t) + lambda*log(cc*etil/alphas/t) );
			}
			sum   += wphi[iphi]                * sumphi;
			sumv1 += wphi[iphi] * cos(tphi)    * sumphi;
			sumv2 += wphi[iphi] * cos(2.*tphi) * sumphi;
			sumv3 += wphi[iphi] * cos(3.*tphi) * sumphi;
		}

		double tcoef = coefBI * f->getDx() * f->getDy() * tau * dtau * t * t * pow(gevtofm, 4);
		sum   *= tcoef;
		sumv1 *= tcoef;
		sumv2 *= tcoef;
		sumv3 *= tcoef;
		yield[ic] += sum;
		v1[ic]    += sumv1;
		v2[ic]    += sumv2;
		v3[ic]    += sumv3;
	}
}

void PhotonsQGPCUDA::addCellBISymm(double tau, double dtau, Fluid *f, EoS *eos, Cell *c) {
	double e, p, nb, nq, ns, t, mub, muq, mus, vx, vy, vz, s, Q[7], tauP;
	tauP = c->getTauP();
	f->getCMFvariables(c, tau, e, nb, nq, ns, vx, vy, vz);
	eos->eos(e, nb, nq, ns, t, mub, muq, mus, p, tauP);
	double lambda = 1.;
	if (taus>0.0) lambda = 1. - exp((tau0-tauP)/taus);
	//s = eos->s(e, nb, nq, ns, tauP);

	if (t<Tcut) return;

	// Sum over all pT values
	for(int ic = 0; ic < pts.size(); ++ic) {
		double pt = pts[ic];
		double sum = 0., sumv1 = 0., sumv2 = 0., sumv3 = 0.;
		
		double sumphi = 0.;
		// Integral over eta
		for(int ieta = 0; ieta < xeta.size(); ++ieta) {
			double etil = (pt * cosh(xeta[ieta]) - vx * pt) / sqrt(1. - vx*vx - vy*vy);
			sumphi += weta[ieta] * exp(-etil / t) * ( lambda*lambda*(log(a*etil/alphas/t) + b*etil/t) + lambda*log(cc*etil/alphas/t) );
		}
		sum   += 2. * C_PI * sumphi;
		sumv1 += 0.;
		sumv2 += 0.;
		sumv3 += 0.;

		double tcoef = coefBI * f->getDx() * f->getDy() * tau * dtau * t * t * pow(gevtofm, 4);
		//double tcoef = coefBI * tau * dtau * t * t * pow(gevtofm, 4);
		sum   *= tcoef;
		sumv1 *= tcoef;
		sumv2 *= tcoef;
		sumv3 *= tcoef;
		yield[ic] += sum;
		v1[ic]    += sumv1;
		v2[ic]    += sumv2;
		v3[ic]    += sumv3;
	}
}

//consts: pT, a, b, cc, alphas
//4-vector: (xphi, cosphi, sinphi, wphi)x32
//4-vector: (xeta, coseta, sineta, weta)x32
//4-vector: (vx, vy, vz, T)
//4-vector: (lambda, lambda, lambda, lambda)

void PhotonsQGPCUDA::addCellsBI(double tau, double dtau, Fluid *f, EoS *eos) {
	int nx = f->getNX();
	int ny = f->getNY();
	int nz = f->getNZ();

	int index = 0;

	for (int ix = 2; ix < nx - 2; ix++)
			for (int iy = 2; iy < ny - 2; iy++) {
				Cell *c = f->getCell(ix, iy, nz / 2);
				if (fCUDA==0) addCellBI(tau, dtau, f, eos, c);
				else {
					double e, p, nb, nq, ns, t, mub, muq, mus, vx, vy, vz, s, Q[7], tauP;
					tauP = c->getTauP();
					f->getCMFvariables(c, tau, e, nb, nq, ns, vx, vy, vz);
					eos->eos(e, nb, nq, ns, t, mub, muq, mus, p, tauP);
					double lambda = 1.;
					if (taus>0.0) lambda = 1. - exp((tau0-tauP)/taus);
					h_vT[index    ] = vx;
					h_vT[index + 1] = vy;
					h_vT[index + 2] = tanh(vz);
					h_vT[index + 3] = t;
					h_lambda[index] = h_lambda[index + 1] = h_lambda[index + 2] = h_lambda[index + 3] = lambda;
					index += 4;
				}
			}

	if (fCUDA!=0) {
	
		cuda_copyArrayToDevice(d_vT, h_vT, fN);
		cuda_copyArrayToDevice(d_lambda, h_lambda, fN);

		for(int ic = 0; ic < pts.size(); ++ic) {
			cuda_addCellsBI_PhotonsQGP(d_phi, d_eta, d_vT, d_lambda, d_res, fN, pts[ic], a, b, cc, alphas, 256, 1);

			cuda_copyArrayFromDevice(h_res, d_res, 0, fN);

			for(int i=0;i<fN;++i) {
				double t = h_vT[4*i+3];
				if (t<Tcut) continue;
				double sum = 0., sumv1 = 0., sumv2 = 0., sumv3 = 0.;
				sum   = h_res[4*i];
				sumv1 = h_res[4*i+1];
				sumv2 = h_res[4*i+2];
				sumv3 = h_res[4*i+3];
				double tcoef = coefBI * f->getDx() * f->getDy() * tau * dtau * t * t * pow(gevtofm, 4);
				sum   *= tcoef;
				sumv1 *= tcoef;
				sumv2 *= tcoef;
				sumv3 *= tcoef;
				yield[ic] += sum;
				v1[ic]    += sumv1;
				v2[ic]    += sumv2;
				v3[ic]    += sumv3;
			}
		}
	}
}

void PhotonsQGPCUDA::addCellsBISymm(double tau, double dtau, Fluid *f, EoS *eos) {
	int nx = f->getNX();
	int ny = f->getNY();
	int nz = f->getNZ();

	int index = 0;

	for (int ix = 2; ix < nx - 2; ix++)
			for (int iy = 2; iy < ny - 2; iy++) {
				Cell *c = f->getCell(ix, iy, nz / 2);
				if (fCUDA==0) addCellBISymm(tau, dtau, f, eos, c);
				else {
					double e, p, nb, nq, ns, t, mub, muq, mus, vx, vy, vz, s, Q[7], tauP;
					tauP = c->getTauP();
					f->getCMFvariables(c, tau, e, nb, nq, ns, vx, vy, vz);
					eos->eos(e, nb, nq, ns, t, mub, muq, mus, p, tauP);
					double lambda = 1.;
					if (taus>0.0) lambda = 1. - exp((tau0-tauP)/taus);
					h_vT[index    ] = vx;
					h_vT[index + 1] = vy;
					h_vT[index + 2] = tanh(vz);
					h_vT[index + 3] = t;
					h_lambda[index] = h_lambda[index + 1] = h_lambda[index + 2] = h_lambda[index + 3] = lambda;
					index += 4;
				}
			}

	if (fCUDA!=0) {

		cuda_copyArrayToDevice(d_vT, h_vT, fN);
		cuda_copyArrayToDevice(d_lambda, h_lambda, fN);

		for(int ic = 0; ic < pts.size(); ++ic) {
			cuda_addCellsBISymm_PhotonsQGP(d_eta, d_vT, d_lambda, d_res, fN, pts[ic], a, b, cc, alphas, 256, 1);

			cuda_copyArrayFromDevice(h_res, d_res, 0, fN);

			for(int i=0;i<fN;++i) {
				double t = h_vT[4*i+3];
				if (t<Tcut) continue;
				double sum = 0., sumv1 = 0., sumv2 = 0., sumv3 = 0.;
				sum   = h_res[4*i];
				sumv1 = h_res[4*i+1];
				sumv2 = h_res[4*i+2];
				sumv3 = h_res[4*i+3];
				double tcoef = coefBI * f->getDx() * f->getDy() * tau * dtau * t * t * pow(gevtofm, 4);
				sum   *= tcoef;
				sumv1 *= tcoef;
				sumv2 *= tcoef;
				sumv3 *= tcoef;
				yield[ic] += sum;
				v1[ic]    += sumv1;
				v2[ic]    += sumv2;
				v3[ic]    += sumv3;
			}
		}
	}
}

// QGP-AMY

PhotonsAMYCUDA::PhotonsAMYCUDA(const std::vector<double> & ptin, const std::vector<double> & yin, double tau0_, double taus_, double Tcut_) : Photons(ptin, yin, Tcut_), tau0(tau0_), taus(taus_)
{
	fN = 0;
	fMode = 0;
}

void PhotonsAMYCUDA::init() {
	A1  = 1. / 6. / C_PI / C_PI;
	A2  = 1. / 3. / C_PI / C_PI;
	B1  = 1.000;
	B2  = 0.112;
	Nf  = 3.;
	Tc  = 0.170;
	Fq  = 2. / 3.;
	aEM = 7.297352e-3;
	
	//coef = 7.297352e-3 * 0.3 / pow(C_PI, 3.) / 4. * (2. / 3.);
	//coefBI = coef * 2.;
	//a = 0.197; b = 0.987; cc = 0.884;
	//a = 0.232; b = 0.987; cc = 0.884;
	//alphas = 0.3;

	GetCoefsIntegrateLegendre32(0., 2.*C_PI, xphi, wphi);
	GetCoefsIntegrateLaguerre32(xeta, weta);
}

void PhotonsAMYCUDA::initCUDAArrays(Fluid *f) {
	fCUDA = 0;
#ifdef USE_CUDA_TOOLKIT
	if (f!=NULL) {
		fN = (f->getNX()-4) * (f->getNY()-4) * (f->getNZ()-4);
		h_vT     = new float[4 * fN];
		h_lambda = new float[4 * fN];
		h_res    = new float[4 * fN];
		h_phi    = new float[32 * 4];
		h_eta    = new float[32 * 4];

		for(int i = 0; i < 32; ++i) {
			double tphi = xphi[i];
			h_phi[4*i]     = (float)tphi;
			h_phi[4*i + 1] = (float)cos(tphi);
			h_phi[4*i + 2] = (float)sin(tphi);
			h_phi[4*i + 3] = (float)wphi[i];

			double teta = xeta[i];
			h_eta[4*i]     = (float)teta;
			h_eta[4*i + 1] = (float)cosh(teta);
			h_eta[4*i + 2] = (float)sinh(teta);
			h_eta[4*i + 3] = (float)weta[i];

			printf("%20E%20E%20E%20E\n", h_eta[4*i], h_eta[4*i+1], h_eta[4*i+2], h_eta[4*i+3]);
		}

		cuda_allocateArray(&d_vT, fN);
		cuda_allocateArray(&d_lambda, fN);
		cuda_allocateArray(&d_res, fN);
		cuda_allocateArray(&d_phi, 32);
		cuda_allocateArray(&d_eta, 32);

		cuda_copyArrayToDevice(d_phi, h_phi, 32);
		cuda_copyArrayToDevice(d_eta, h_eta, 32);
		
		fCUDA = 1;
	}
#endif
}

PhotonsAMYCUDA::~PhotonsAMYCUDA() {
	if (fCUDA==1) {
		delete [] h_vT;
		delete [] h_lambda;
		delete [] h_res;
		delete [] h_phi;
		delete [] h_eta;
		cuda_deleteArray(d_vT);
		cuda_deleteArray(d_lambda);
		cuda_deleteArray(d_res);
		cuda_deleteArray(d_phi);
		cuda_deleteArray(d_eta);
		//cuda_threadExit();
	}
}

/*
void PhotonsAMYCUDA::addCell(double tau, double dtau, Fluid *f, EoS *eos, Cell *c) {
	double e, p, nb, nq, ns, t, mub, muq, mus, vx, vy, vz, s, Q[7], tauP;
	tauP = c->getTauP();
	f->getCMFvariables(c, tau, e, nb, nq, ns, vx, vy, vz);
	eos->eos(e, nb, nq, ns, t, mub, muq, mus, p, tauP);
	double lambda = 1.;
	if (taus>0.0) lambda = 1. - exp((tau0-tauP)/taus);
	//s = eos->s(e, nb, nq, ns, tauP);

	if (t<Tcut) return;

	// Sum over all pT and Y values
	for(int ic = 0; ic < pts.size(); ++ic) {
		double pt = pts[ic], Y = ys[ic];
		double sum = 0., sumv1 = 0., sumv2 = 0., sumv3 = 0.;
		double eta = c->getZ();
		double gamma = 1. / sqrt(1. - vx*vx - vy*vy - tanh(vz)*tanh(vz));
		double Etil1 = gamma * pt * (cosh(Y) - tanh(vz) * sinh(Y)); 
		// Integral over phi
		for(int iphi = 0; iphi < xphi.size(); ++iphi) {
			double tphi = xphi[iphi];
			double cosphi = cos(tphi), sinphi = sin(tphi);
			double sumphi = 0.;
			double Etil2 = -gamma * pt * (vx*cosphi + vy*sinphi);
			double etil = Etil1 + Etil2;
			//sumphi += exp(-etil / t) * ( lambda*lambda*(log(a*etil/alphas/t) + b*etil/t) + lambda*log(cc*etil/alphas/t) );
			sumphi += GL(etil/t, t, lambda, fMode);
			sum    += wphi[iphi]                * sumphi;
			sumv1  += wphi[iphi] * cos(tphi)    * sumphi;
			sumv2  += wphi[iphi] * cos(2.*tphi) * sumphi;
			sumv3  += wphi[iphi] * cos(3.*tphi) * sumphi;
		}

		//double tcoef = coef * f->getDx() * f->getDy() * f->getDz() * tau * dtau * t * t * pow(gevtofm, 4);
		double tcoef = f->getDx() * f->getDy() * f->getDz() * tau * dtau * pow(gevtofm, 4) / 2. / C_PI;
		sum       *= tcoef;
		sumv1     *= tcoef;
		sumv2     *= tcoef;
		sumv3     *= tcoef;
		yield[ic] += sum;
		v1[ic]    += sumv1;
		v2[ic]    += sumv2;
		v3[ic]    += sumv3;
	}
}*/

void PhotonsAMYCUDA::addCellBI(double tau, double dtau, Fluid *f, EoS *eos, Cell *c) {
	double e, p, nb, nq, ns, t, mub, muq, mus, vx, vy, vz, s, Q[7], tauP;
	tauP = c->getTauP();
	f->getCMFvariables(c, tau, e, nb, nq, ns, vx, vy, vz);
	eos->eos(e, nb, nq, ns, t, mub, muq, mus, p, tauP);
	double lambda = 1.;
	if (taus>0.0) lambda = 1. - exp((tau0-tauP)/taus);
	//s = eos->s(e, nb, nq, ns, tauP);

	if (t<Tcut) return;

	// Sum over all pT values
	for(int ic = 0; ic < pts.size(); ++ic) {
		double pt = pts[ic];
		double sum = 0., sumv1 = 0., sumv2 = 0., sumv3 = 0.;
		// Integral over phi
		for(int iphi = 0; iphi < xphi.size(); ++iphi) {
			double tphi = xphi[iphi];
			double cosphi = cos(tphi), sinphi = sin(tphi);
			double sumphi = 0.;
			// Integral over eta
			for(int ieta = 0; ieta < xeta.size(); ++ieta) {
				double etil = (pt * cosh(xeta[ieta]) - vx * pt * cosphi - vy * pt * sinphi) / sqrt(1. - vx*vx - vy*vy);
				//sumphi += weta[ieta] * exp(-etil / t) * ( lambda*lambda*(log(a*etil/alphas/t) + b*etil/t) + lambda*log(cc*etil/alphas/t) );
				sumphi += weta[ieta] * GL(etil/t, t, lambda, fMode);
			}
			sum   += wphi[iphi]                * sumphi;
			sumv1 += wphi[iphi] * cos(tphi)    * sumphi;
			sumv2 += wphi[iphi] * cos(2.*tphi) * sumphi;
			sumv3 += wphi[iphi] * cos(3.*tphi) * sumphi;
		}

		//double tcoef = coefBI * f->getDx() * f->getDy() * tau * dtau * t * t * pow(gevtofm, 4);
		double tcoef = 2. * f->getDx() * f->getDy() * tau * dtau / 2. / C_PI;
		sum   *= tcoef;
		sumv1 *= tcoef;
		sumv2 *= tcoef;
		sumv3 *= tcoef;
		yield[ic] += sum;
		v1[ic]    += sumv1;
		v2[ic]    += sumv2;
		v3[ic]    += sumv3;
	}
}

void PhotonsAMYCUDA::addCellBISymm(double tau, double dtau, Fluid *f, EoS *eos, Cell *c) {
	double e, p, nb, nq, ns, t, mub, muq, mus, vx, vy, vz, s, Q[7], tauP;
	tauP = c->getTauP();
	f->getCMFvariables(c, tau, e, nb, nq, ns, vx, vy, vz);
	eos->eos(e, nb, nq, ns, t, mub, muq, mus, p, tauP);
	double lambda = 1.;
	if (taus>0.0) lambda = 1. - exp((tau0-tauP)/taus);
	//s = eos->s(e, nb, nq, ns, tauP);

	if (t<Tcut) return;

	// Sum over all pT values
	for(int ic = 0; ic < pts.size(); ++ic) {
		double pt = pts[ic];
		double sum = 0., sumv1 = 0., sumv2 = 0., sumv3 = 0.;
		
		double sumphi = 0.;
		// Integral over eta
		for(int ieta = 0; ieta < xeta.size(); ++ieta) {
			double etil = (pt * cosh(xeta[ieta]) - vx * pt) / sqrt(1. - vx*vx - vy*vy);
			//sumphi += weta[ieta] * exp(-etil / t) * ( lambda*lambda*(log(a*etil/alphas/t) + b*etil/t) + lambda*log(cc*etil/alphas/t) );
			sumphi += weta[ieta] * GL(etil/t, t, lambda, fMode);
		}
		sum   += 2. * C_PI * sumphi;
		sumv1 += 0.;
		sumv2 += 0.;
		sumv3 += 0.;

		//double tcoef = coefBI * f->getDx() * f->getDy() * tau * dtau * t * t * pow(gevtofm, 4);
		//double tcoef = coefBI * tau * dtau * t * t * pow(gevtofm, 4);
		double tcoef = 2. * f->getDx() * f->getDy() * tau * dtau / 2. / C_PI;
		sum   *= tcoef;
		sumv1 *= tcoef;
		sumv2 *= tcoef;
		sumv3 *= tcoef;
		yield[ic] += sum;
		v1[ic]    += sumv1;
		v2[ic]    += sumv2;
		v3[ic]    += sumv3;
	}
}

double PhotonsAMYCUDA::alphas(double T) const {
	return 6. * C_PI / (33. - 2. * Nf) / log(8. * T / Tc);
}

double PhotonsAMYCUDA::G1(double E, double T) const {
	return A1 * Fq * aEM * alphas(T) * T * T * exp(-E) * log(B1*E/alphas(T)) * pow(gevtofm, 4);
}

double PhotonsAMYCUDA::G2(double E, double T) const {
	return A2 * Fq * aEM * alphas(T) * T * T * exp(-E) * log(B2*E/alphas(T)) * pow(gevtofm, 4);
}

double PhotonsAMYCUDA::G(double E, double T) const {
	return 1. / C_PI / C_PI * Fq * aEM * alphas(T) * T * T / (exp(E) + 1) * (0.5 * log(3.*E/2./C_PI/alphas(T)) + C12(E) + C34(E)) * pow(gevtofm, 4);
}

double PhotonsAMYCUDA::C12(double E) const {
	return (0.041 / E) - 0.3615 + 1.01 * exp(-1.35 * E);
}

double PhotonsAMYCUDA::C34(double E) const {
	return sqrt(1. + Nf/6.) * (0.548 / pow(E, 3./2.) * log(12.28 + 1./E) + 0.133 * E / sqrt(1. + E/16.27));
}

double PhotonsAMYCUDA::GL(double E, double T,  double lambda, int mode) const {
	//return 4. / 3. / pow(C_PI, 4.) * aEM * 0.3 * T * T * exp(-E) * (2. * log(0.417*E/0.3) + 0.987 * E);
	if (mode==0) return lambda * G1(E,T) + lambda * lambda * (G(E,T) - G1(E,T));
	else return lambda * lambda * G2(E,T) + lambda * (G(E,T) - G2(E,T));
}


void PhotonsAMYCUDA::addCellsBI(double tau, double dtau, Fluid *f, EoS *eos) {
	int nx = f->getNX();
	int ny = f->getNY();
	int nz = f->getNZ();

	int index = 0;

	for (int ix = 2; ix < nx - 2; ix++)
			for (int iy = 2; iy < ny - 2; iy++) {
				Cell *c = f->getCell(ix, iy, nz / 2);
				if (fCUDA==0) addCellBI(tau, dtau, f, eos, c);
				else {
					double e, p, nb, nq, ns, t, mub, muq, mus, vx, vy, vz, s, Q[7], tauP;
					tauP = c->getTauP();
					f->getCMFvariables(c, tau, e, nb, nq, ns, vx, vy, vz);
					eos->eos(e, nb, nq, ns, t, mub, muq, mus, p, tauP);
					double lambda = 1.;
					if (taus>0.0) lambda = 1. - exp((tau0-tauP)/taus);
					h_vT[index    ] = vx;
					h_vT[index + 1] = vy;
					h_vT[index + 2] = tanh(vz);
					h_vT[index + 3] = t;
					h_lambda[index] = h_lambda[index + 1] = h_lambda[index + 2] = h_lambda[index + 3] = lambda;
					h_lambda[index + 1] = alphas(t);
					index += 4;
				}
			}

	if (fCUDA!=0) {
	
		cuda_copyArrayToDevice(d_vT, h_vT, fN);
		cuda_copyArrayToDevice(d_lambda, h_lambda, fN);

		for(int ic = 0; ic < pts.size(); ++ic) {
			if (fMode==1) cuda_addCellsBI_PhotonsAMYHigh(d_phi, d_eta, d_vT, d_lambda, d_res, fN, pts[ic], A1, A2, B1, B2, Fq, Nf, aEM, 256, 1);
			else cuda_addCellsBI_PhotonsAMYLow(d_phi, d_eta, d_vT, d_lambda, d_res, fN, pts[ic], A1, A2, B1, B2, Fq, Nf, aEM, 256, 1);

			cuda_copyArrayFromDevice(h_res, d_res, 0, fN);

			for(int i=0;i<fN;++i) {
				double t = h_vT[4*i+3];
				if (t<Tcut) continue;
				double sum = 0., sumv1 = 0., sumv2 = 0., sumv3 = 0.;
				sum   = h_res[4*i];
				sumv1 = h_res[4*i+1];
				sumv2 = h_res[4*i+2];
				sumv3 = h_res[4*i+3];
				double tcoef = 2. * f->getDx() * f->getDy() * tau * dtau / 2. / C_PI;
				sum   *= tcoef;
				sumv1 *= tcoef;
				sumv2 *= tcoef;
				sumv3 *= tcoef;
				yield[ic] += sum;
				v1[ic]    += sumv1;
				v2[ic]    += sumv2;
				v3[ic]    += sumv3;
			}
		}
	}
}

void PhotonsAMYCUDA::addCellsBISymm(double tau, double dtau, Fluid *f, EoS *eos) {
	int nx = f->getNX();
	int ny = f->getNY();
	int nz = f->getNZ();

	int index = 0;

	for (int ix = 2; ix < nx - 2; ix++)
			for (int iy = 2; iy < ny - 2; iy++) {
				Cell *c = f->getCell(ix, iy, nz / 2);
				if (fCUDA==0) addCellBISymm(tau, dtau, f, eos, c);
				else {
					double e, p, nb, nq, ns, t, mub, muq, mus, vx, vy, vz, s, Q[7], tauP;
					tauP = c->getTauP();
					f->getCMFvariables(c, tau, e, nb, nq, ns, vx, vy, vz);
					eos->eos(e, nb, nq, ns, t, mub, muq, mus, p, tauP);
					double lambda = 1.;
					if (taus>0.0) lambda = 1. - exp((tau0-tauP)/taus);
					h_vT[index    ] = vx;
					h_vT[index + 1] = vy;
					h_vT[index + 2] = tanh(vz);
					h_vT[index + 3] = t;
					h_lambda[index] = h_lambda[index + 1] = h_lambda[index + 2] = h_lambda[index + 3] = lambda;
					h_lambda[index + 1] = alphas(t);
					index += 4;
				}
			}

	if (fCUDA!=0) {

		cuda_copyArrayToDevice(d_vT, h_vT, fN);
		cuda_copyArrayToDevice(d_lambda, h_lambda, fN);

		for(int ic = 0; ic < pts.size(); ++ic) {
			if (fMode==1) cuda_addCellsBISymm_PhotonsAMYHigh(d_eta, d_vT, d_lambda, d_res, fN, pts[ic], A1, A2, B1, B2, Fq, Nf, aEM, 256, 1);
			else cuda_addCellsBISymm_PhotonsAMYLow(d_eta, d_vT, d_lambda, d_res, fN, pts[ic], A1, A2, B1, B2, Fq, Nf, aEM, 256, 1);

			cuda_copyArrayFromDevice(h_res, d_res, 0, fN);

			for(int i=0;i<fN;++i) {
				double t = h_vT[4*i+3];
				if (t<Tcut) continue;
				double sum = 0., sumv1 = 0., sumv2 = 0., sumv3 = 0.;
				sum   = h_res[4*i];
				sumv1 = h_res[4*i+1];
				sumv2 = h_res[4*i+2];
				sumv3 = h_res[4*i+3];
				double tcoef = 2. * f->getDx() * f->getDy() * tau * dtau * t * t;
				sum   *= tcoef;
				sumv1 *= tcoef;
				sumv2 *= tcoef;
				sumv3 *= tcoef;
				yield[ic] += sum;
				v1[ic]    += sumv1;
				v2[ic]    += sumv2;
				v3[ic]    += sumv3;
			}
		}
	}
}




// HADRONS

PhotonsHadronsCUDA::PhotonsHadronsCUDA(const std::vector<double> & ptin, const std::vector<double> & yin, double tau0_, double taus_, double Tcut_) : Photons(ptin, yin, Tcut_), tau0(tau0_), taus(taus_)
{
	fN = 0;
	fMode = 0;
}

void PhotonsHadronsCUDA::init() {
	GetCoefsIntegrateLegendre32(0., 2.*C_PI, xphi, wphi);
	GetCoefsIntegrateLaguerre32(xeta, weta);
}

void PhotonsHadronsCUDA::initCUDAArrays(Fluid *f) {
	fCUDA = 0;
#ifdef USE_CUDA_TOOLKIT
	if (f!=NULL) {
		fN = (f->getNX()-4) * (f->getNY()-4) * (f->getNZ()-4);
		h_vT     = new float[4 * fN];
		h_lambda = new float[4 * fN];
		h_res    = new float[4 * fN];
		h_phi    = new float[32 * 4];
		h_eta    = new float[32 * 4];

		for(int i = 0; i < 32; ++i) {
			double tphi = xphi[i];
			h_phi[4*i]     = (float)tphi;
			h_phi[4*i + 1] = (float)cos(tphi);
			h_phi[4*i + 2] = (float)sin(tphi);
			h_phi[4*i + 3] = (float)wphi[i];

			double teta = xeta[i];
			h_eta[4*i]     = (float)teta;
			h_eta[4*i + 1] = (float)cosh(teta);
			h_eta[4*i + 2] = (float)sinh(teta);
			h_eta[4*i + 3] = (float)weta[i];

			printf("%20E%20E%20E%20E\n", h_eta[4*i], h_eta[4*i+1], h_eta[4*i+2], h_eta[4*i+3]);
		}

		cuda_allocateArray(&d_vT, fN);
		cuda_allocateArray(&d_lambda, fN);
		cuda_allocateArray(&d_res, fN);
		cuda_allocateArray(&d_phi, 32);
		cuda_allocateArray(&d_eta, 32);

		cuda_copyArrayToDevice(d_phi, h_phi, 32);
		cuda_copyArrayToDevice(d_eta, h_eta, 32);
		
		fCUDA = 1;
	}
#endif
}

PhotonsHadronsCUDA::~PhotonsHadronsCUDA() {
	if (fCUDA==1) {
		delete [] h_vT;
		delete [] h_lambda;
		delete [] h_res;
		delete [] h_phi;
		delete [] h_eta;
		cuda_deleteArray(d_vT);
		cuda_deleteArray(d_lambda);
		cuda_deleteArray(d_res);
		cuda_deleteArray(d_phi);
		cuda_deleteArray(d_eta);
		//cuda_threadExit();
	}
}

void PhotonsHadronsCUDA::addCellBI(double tau, double dtau, Fluid *f, EoS *eos, Cell *c) {
	double e, p, nb, nq, ns, t, mub, muq, mus, vx, vy, vz, s, Q[7], tauP;
	tauP = c->getTauP();
	f->getCMFvariables(c, tau, e, nb, nq, ns, vx, vy, vz);
	eos->eos(e, nb, nq, ns, t, mub, muq, mus, p, tauP);
	double lambda = 1.;
	if (taus>0.0) lambda = 1. - exp((tau0-tauP)/taus);
	//s = eos->s(e, nb, nq, ns, tauP);

	if (t<Tcut) return;

	// Sum over all pT values
	for(int ic = 0; ic < pts.size(); ++ic) {
		double pt = pts[ic];
		double sum = 0., sumv1 = 0., sumv2 = 0., sumv3 = 0.;
		// Integral over phi
		for(int iphi = 0; iphi < xphi.size(); ++iphi) {
			double tphi = xphi[iphi];
			double cosphi = cos(tphi), sinphi = sin(tphi);
			double sumphi = 0.;
			// Integral over eta
			for(int ieta = 0; ieta < xeta.size(); ++ieta) {
				double etil = (pt * cosh(xeta[ieta]) - vx * pt * cosphi - vy * pt * sinphi) / sqrt(1. - vx*vx - vy*vy);
				//sumphi += weta[ieta] * exp(-etil / t) * ( lambda*lambda*(log(a*etil/alphas/t) + b*etil/t) + lambda*log(cc*etil/alphas/t) );
				//sumphi += weta[ieta] * GL(etil/t, t, lambda, fMode);
				sumphi += weta[ieta] * PPR(etil, t);
			}
			sum   += wphi[iphi]                * sumphi;
			sumv1 += wphi[iphi] * cos(tphi)    * sumphi;
			sumv2 += wphi[iphi] * cos(2.*tphi) * sumphi;
			sumv3 += wphi[iphi] * cos(3.*tphi) * sumphi;
		}

		//double tcoef = coefBI * f->getDx() * f->getDy() * tau * dtau * t * t * pow(gevtofm, 4);
		double tcoef = 2. * f->getDx() * f->getDy() * tau * dtau / 2. / C_PI;
		sum   *= tcoef;
		sumv1 *= tcoef;
		sumv2 *= tcoef;
		sumv3 *= tcoef;
		yield[ic] += sum;
		v1[ic]    += sumv1;
		v2[ic]    += sumv2;
		v3[ic]    += sumv3;
	}
}

void PhotonsHadronsCUDA::addCellBISymm(double tau, double dtau, Fluid *f, EoS *eos, Cell *c) {
	double e, p, nb, nq, ns, t, mub, muq, mus, vx, vy, vz, s, Q[7], tauP;
	tauP = c->getTauP();
	f->getCMFvariables(c, tau, e, nb, nq, ns, vx, vy, vz);
	eos->eos(e, nb, nq, ns, t, mub, muq, mus, p, tauP);
	double lambda = 1.;
	if (taus>0.0) lambda = 1. - exp((tau0-tauP)/taus);
	//s = eos->s(e, nb, nq, ns, tauP);

	if (t<Tcut) return;

	// Sum over all pT values
	for(int ic = 0; ic < pts.size(); ++ic) {
		double pt = pts[ic];
		double sum = 0., sumv1 = 0., sumv2 = 0., sumv3 = 0.;
		
		double sumphi = 0.;
		// Integral over eta
		for(int ieta = 0; ieta < xeta.size(); ++ieta) {
			double etil = (pt * cosh(xeta[ieta]) - vx * pt) / sqrt(1. - vx*vx - vy*vy);
			//sumphi += weta[ieta] * exp(-etil / t) * ( lambda*lambda*(log(a*etil/alphas/t) + b*etil/t) + lambda*log(cc*etil/alphas/t) );
			//sumphi += weta[ieta] * GL(etil/t, t, lambda, fMode);
			sumphi += weta[ieta] * PPR(etil, t);
		}
		sum   += 2. * C_PI * sumphi;
		sumv1 += 0.;
		sumv2 += 0.;
		sumv3 += 0.;

		//double tcoef = coefBI * f->getDx() * f->getDy() * tau * dtau * t * t * pow(gevtofm, 4);
		//double tcoef = coefBI * tau * dtau * t * t * pow(gevtofm, 4);
		double tcoef = 2. * f->getDx() * f->getDy() * tau * dtau / 2. / C_PI;
		sum   *= tcoef;
		sumv1 *= tcoef;
		sumv2 *= tcoef;
		sumv3 *= tcoef;
		yield[ic] += sum;
		v1[ic]    += sumv1;
		v2[ic]    += sumv2;
		v3[ic]    += sumv3;
	}
}


void PhotonsHadronsCUDA::addCellsBI(double tau, double dtau, Fluid *f, EoS *eos) {
	int nx = f->getNX();
	int ny = f->getNY();
	int nz = f->getNZ();

	int index = 0;

	for (int ix = 2; ix < nx - 2; ix++)
			for (int iy = 2; iy < ny - 2; iy++) {
				Cell *c = f->getCell(ix, iy, nz / 2);
				if (fCUDA==0) addCellBI(tau, dtau, f, eos, c);
				else {
					double e, p, nb, nq, ns, t, mub, muq, mus, vx, vy, vz, s, Q[7], tauP;
					tauP = c->getTauP();
					f->getCMFvariables(c, tau, e, nb, nq, ns, vx, vy, vz);
					eos->eos(e, nb, nq, ns, t, mub, muq, mus, p, tauP);
					double lambda = 1.;
					if (taus>0.0) lambda = 1. - exp((tau0-tauP)/taus);
					h_vT[index    ] = vx;
					h_vT[index + 1] = vy;
					h_vT[index + 2] = tanh(vz);
					h_vT[index + 3] = t;
					h_lambda[index] = h_lambda[index + 1] = h_lambda[index + 2] = h_lambda[index + 3] = lambda;
					index += 4;
				}
			}

	if (fCUDA!=0) {
	
		cuda_copyArrayToDevice(d_vT, h_vT, fN);
		cuda_copyArrayToDevice(d_lambda, h_lambda, fN);

		for(int ic = 0; ic < pts.size(); ++ic) {
			cuda_addCellsBI_PhotonsHadrons(d_phi, d_eta, d_vT, d_lambda, d_res, fN, pts[ic], 256, 1);

			cuda_copyArrayFromDevice(h_res, d_res, 0, fN);

			for(int i=0;i<fN;++i) {
				double t = h_vT[4*i+3];
				if (t<Tcut) continue;
				double sum = 0., sumv1 = 0., sumv2 = 0., sumv3 = 0.;
				sum   = h_res[4*i];
				sumv1 = h_res[4*i+1];
				sumv2 = h_res[4*i+2];
				sumv3 = h_res[4*i+3];
				double tcoef = 2. * f->getDx() * f->getDy() * tau * dtau / 2. / C_PI;
				sum   *= tcoef;
				sumv1 *= tcoef;
				sumv2 *= tcoef;
				sumv3 *= tcoef;
				yield[ic] += sum;
				v1[ic]    += sumv1;
				v2[ic]    += sumv2;
				v3[ic]    += sumv3;
			}
		}
	}
}

void PhotonsHadronsCUDA::addCellsBISymm(double tau, double dtau, Fluid *f, EoS *eos) {
	int nx = f->getNX();
	int ny = f->getNY();
	int nz = f->getNZ();

	int index = 0;

	for (int ix = 2; ix < nx - 2; ix++)
			for (int iy = 2; iy < ny - 2; iy++) {
				Cell *c = f->getCell(ix, iy, nz / 2);
				if (fCUDA==0) addCellBISymm(tau, dtau, f, eos, c);
				else {
					double e, p, nb, nq, ns, t, mub, muq, mus, vx, vy, vz, s, Q[7], tauP;
					tauP = c->getTauP();
					f->getCMFvariables(c, tau, e, nb, nq, ns, vx, vy, vz);
					eos->eos(e, nb, nq, ns, t, mub, muq, mus, p, tauP);
					double lambda = 1.;
					if (taus>0.0) lambda = 1. - exp((tau0-tauP)/taus);
					h_vT[index    ] = vx;
					h_vT[index + 1] = vy;
					h_vT[index + 2] = tanh(vz);
					h_vT[index + 3] = t;
					h_lambda[index] = h_lambda[index + 1] = h_lambda[index + 2] = h_lambda[index + 3] = lambda;
					index += 4;
				}
			}

	if (fCUDA!=0) {

		cuda_copyArrayToDevice(d_vT, h_vT, fN);
		cuda_copyArrayToDevice(d_lambda, h_lambda, fN);

		for(int ic = 0; ic < pts.size(); ++ic) {
			cuda_addCellsBISymm_PhotonsHadrons(d_eta, d_vT, d_lambda, d_res, fN, pts[ic], 256, 1);

			cuda_copyArrayFromDevice(h_res, d_res, 0, fN);

			for(int i=0;i<fN;++i) {
				double t = h_vT[4*i+3];
				if (t<Tcut) continue;
				double sum = 0., sumv1 = 0., sumv2 = 0., sumv3 = 0.;
				sum   = h_res[4*i];
				sumv1 = h_res[4*i+1];
				sumv2 = h_res[4*i+2];
				sumv3 = h_res[4*i+3];
				double tcoef = 2. * f->getDx() * f->getDy() * tau * dtau * t * t;
				sum   *= tcoef;
				sumv1 *= tcoef;
				sumv2 *= tcoef;
				sumv3 *= tcoef;
				yield[ic] += sum;
				v1[ic]    += sumv1;
				v2[ic]    += sumv2;
				v3[ic]    += sumv3;
			}
		}
	}
}


double PhotonsHadronsCUDA::PPR_PKK(double E, double T) {
	if (E>20.) return 0.;
	double RPIK   = 1. / pow(T,3.) * exp(-(5.4018*pow(T,-0.6864)-1.51)*pow(2.*T*E, 0.07) - 0.91*E/T);
	double RPIKs  = pow(T,3.75) * exp(-0.35/pow(2.*T*E,1.05) + (2.3894*pow(T,0.03435) - 3.222)*E/T);
	double RPIKsK = pow(T,3.70) * exp(-(6.096*pow(T,1.889)+1.0299)/pow(2.*T*E,-1.613*pow(T,2.162)+0.975) - 0.96*E/T);
	return RPIK + RPIKs + RPIKsK;
}

double PhotonsHadronsCUDA::PPR_rho(double E, double T) {
	if (E>20.) return 0.;
	double aT = -31.21 + 353.61*T - 1739.4*T*T + 3105. * T * T * T;
	double bT = -5.513 - 42.2*T + 333.*T*T - 570.*T*T*T;
	double cT = -6.153 + 57.*T - 134.61*T*T + 8.31*T*T*T;
	return exp(aT*E + bT + cT / (E + 0.2));
}

double PhotonsHadronsCUDA::PPR_brem(double E, double T) {
	if (E>20.) return 0.;
	double abT = -16.28 + 62.45*T - 93.4*T*T - 7.5*T*T*T;
	double bbT = -35.54 + 414.8*T - 2054.*T*T + 3718.8*T*T*T;
	double gbT = 0.7364 - 10.72*T + 56.32*T*T - 103.5*T*T*T;
	double dbT = -2.51 + 58.152*T - 318.24*T*T + 610.7*T*T*T;
	return exp(abT + bbT*E + gbT*E*E + dbT/(E+0.2));
}


// MIX

PhotonsMixCUDA::PhotonsMixCUDA(const std::vector<double> & ptin, const std::vector<double> & yin, double tau0_, double taus_, double Tcut_, double Tc_) : Photons(ptin, yin, Tcut_), tau0(tau0_), taus(taus_), Tsw(Tc_)
{
	fN = 0;
	fMode = 0;
}

void PhotonsMixCUDA::init() {
	A1  = 1. / 6. / C_PI / C_PI;
	A2  = 1. / 3. / C_PI / C_PI;
	B1  = 1.000;
	B2  = 0.112;
	Nf  = 3.;
	TcAMY  = 0.170;
	Fq  = 2. / 3.;
	aEM = 7.297352e-3;

	GetCoefsIntegrateLegendre32(0., 2.*C_PI, xphi, wphi);
	GetCoefsIntegrateLaguerre32(xeta, weta);
}

void PhotonsMixCUDA::initCUDAArrays(Fluid *f) {
	fCUDA = 0;
#ifdef USE_CUDA_TOOLKIT
	if (f!=NULL) {
		fN = (f->getNX()-4) * (f->getNY()-4) * (f->getNZ()-4);
		h_vT     = new float[4 * fN];
		h_lambda = new float[4 * fN];
		h_res    = new float[4 * fN];
		h_phi    = new float[32 * 4];
		h_eta    = new float[32 * 4];

		for(int i = 0; i < 32; ++i) {
			double tphi = xphi[i];
			h_phi[4*i]     = (float)tphi;
			h_phi[4*i + 1] = (float)cos(tphi);
			h_phi[4*i + 2] = (float)sin(tphi);
			h_phi[4*i + 3] = (float)wphi[i];

			double teta = xeta[i];
			h_eta[4*i]     = (float)teta;
			h_eta[4*i + 1] = (float)cosh(teta);
			h_eta[4*i + 2] = (float)sinh(teta);
			h_eta[4*i + 3] = (float)weta[i];

			printf("%20E%20E%20E%20E\n", h_eta[4*i], h_eta[4*i+1], h_eta[4*i+2], h_eta[4*i+3]);
		}

		cuda_allocateArray(&d_vT, fN);
		cuda_allocateArray(&d_lambda, fN);
		cuda_allocateArray(&d_res, fN);
		cuda_allocateArray(&d_phi, 32);
		cuda_allocateArray(&d_eta, 32);

		cuda_copyArrayToDevice(d_phi, h_phi, 32);
		cuda_copyArrayToDevice(d_eta, h_eta, 32);
		
		fCUDA = 1;
	}
#endif
}

PhotonsMixCUDA::~PhotonsMixCUDA() {
	if (fCUDA==1) {
		delete [] h_vT;
		delete [] h_lambda;
		delete [] h_res;
		delete [] h_phi;
		delete [] h_eta;
		cuda_deleteArray(d_vT);
		cuda_deleteArray(d_lambda);
		cuda_deleteArray(d_res);
		cuda_deleteArray(d_phi);
		cuda_deleteArray(d_eta);
		//cuda_threadExit();
	}
}

void PhotonsMixCUDA::addCellBI(double tau, double dtau, Fluid *f, EoS *eos, Cell *c) {

	double e, p, nb, nq, ns, t, mub, muq, mus, vx, vy, vz, s, Q[7], tauP;
	tauP = c->getTauP();
	f->getCMFvariables(c, tau, e, nb, nq, ns, vx, vy, vz);
	eos->eos(e, nb, nq, ns, t, mub, muq, mus, p, tauP);
	double lambda = 1.;
	if (taus>0.0) lambda = 1. - exp((tau0-tauP)/taus);
	//s = eos->s(e, nb, nq, ns, tauP);

	if (t<Tcut) return;
	// printf("PhotonsMixCUDA::addCellBI\n");
	// Sum over all pT values
	for(int ic = 0; ic < pts.size(); ++ic) {
		double pt = pts[ic];
		double sum = 0., sumv1 = 0., sumv2 = 0., sumv3 = 0.;
		// Integral over phi
		for(int iphi = 0; iphi < xphi.size(); ++iphi) {
			double tphi = xphi[iphi];
			double cosphi = cos(tphi), sinphi = sin(tphi);
			double sumphi = 0.;
			// Integral over eta
			for(int ieta = 0; ieta < xeta.size(); ++ieta) {
				double etil = (pt * cosh(xeta[ieta]) - vx * pt * cosphi - vy * pt * sinphi) / sqrt(1. - vx*vx - vy*vy);
				//sumphi += weta[ieta] * exp(-etil / t) * ( lambda*lambda*(log(a*etil/alphas/t) + b*etil/t) + lambda*log(cc*etil/alphas/t) );
				if (t>Tsw) sumphi += weta[ieta] * GL(etil/t, t, lambda, fMode);
				else sumphi += weta[ieta] * PPR(etil, t);
			}
			sum   += wphi[iphi]                * sumphi;
			sumv1 += wphi[iphi] * cos(tphi)    * sumphi;
			sumv2 += wphi[iphi] * cos(2.*tphi) * sumphi;
			sumv3 += wphi[iphi] * cos(3.*tphi) * sumphi;
		}

		//double tcoef = coefBI * f->getDx() * f->getDy() * tau * dtau * t * t * pow(gevtofm, 4);
		double tcoef = 2. * f->getDx() * f->getDy() * tau * dtau / 2. / C_PI;
		sum   *= tcoef;
		sumv1 *= tcoef;
		sumv2 *= tcoef;
		sumv3 *= tcoef;
		yield[ic] += sum;
		v1[ic]    += sumv1;
		v2[ic]    += sumv2;
		v3[ic]    += sumv3;

		// printf("%20E%20E%\n", sum, tcoef);
	}
}

void PhotonsMixCUDA::addCellBISymm(double tau, double dtau, Fluid *f, EoS *eos, Cell *c) {
	double e, p, nb, nq, ns, t, mub, muq, mus, vx, vy, vz, s, Q[7], tauP;
	tauP = c->getTauP();
	f->getCMFvariables(c, tau, e, nb, nq, ns, vx, vy, vz);
	eos->eos(e, nb, nq, ns, t, mub, muq, mus, p, tauP);
	double lambda = 1.;
	if (taus>0.0) lambda = 1. - exp((tau0-tauP)/taus);
	//s = eos->s(e, nb, nq, ns, tauP);

	if (t<Tcut) return;

	// Sum over all pT values
	for(int ic = 0; ic < pts.size(); ++ic) {
		double pt = pts[ic];
		double sum = 0., sumv1 = 0., sumv2 = 0., sumv3 = 0.;
		
		double sumphi = 0.;
		// Integral over eta
		for(int ieta = 0; ieta < xeta.size(); ++ieta) {
			double etil = (pt * cosh(xeta[ieta]) - vx * pt) / sqrt(1. - vx*vx - vy*vy);
			//sumphi += weta[ieta] * exp(-etil / t) * ( lambda*lambda*(log(a*etil/alphas/t) + b*etil/t) + lambda*log(cc*etil/alphas/t) );
			if (t>TcAMY) sumphi += weta[ieta] * GL(etil/t, t, lambda, fMode);
			else sumphi += weta[ieta] * PPR(etil, t);
		}
		sum   += 2. * C_PI * sumphi;
		sumv1 += 0.;
		sumv2 += 0.;
		sumv3 += 0.;

		//double tcoef = coefBI * f->getDx() * f->getDy() * tau * dtau * t * t * pow(gevtofm, 4);
		//double tcoef = coefBI * tau * dtau * t * t * pow(gevtofm, 4);
		double tcoef = 2. * f->getDx() * f->getDy() * tau * dtau / 2. / C_PI;
		sum   *= tcoef;
		sumv1 *= tcoef;
		sumv2 *= tcoef;
		sumv3 *= tcoef;
		yield[ic] += sum;
		v1[ic]    += sumv1;
		v2[ic]    += sumv2;
		v3[ic]    += sumv3;
	}
}

void PhotonsMixCUDA::addCellsBI(double tau, double dtau, Fluid *f, EoS *eos) {
	int nx = f->getNX();
	int ny = f->getNY();
	int nz = f->getNZ();

	int index = 0;

	for (int ix = 2; ix < nx - 2; ix++)
			for (int iy = 2; iy < ny - 2; iy++) {
				Cell *c = f->getCell(ix, iy, nz / 2);
				if (fCUDA==0) addCellBI(tau, dtau, f, eos, c);
				else {
					double e, p, nb, nq, ns, t, mub, muq, mus, vx, vy, vz, s, Q[7], tauP;
					tauP = c->getTauP();
					f->getCMFvariables(c, tau, e, nb, nq, ns, vx, vy, vz);
					eos->eos(e, nb, nq, ns, t, mub, muq, mus, p, tauP);
					double lambda = 1.;
					if (taus>0.0) lambda = 1. - exp((tau0-tauP)/taus);
					h_vT[index    ] = vx;
					h_vT[index + 1] = vy;
					h_vT[index + 2] = tanh(vz);
					h_vT[index + 3] = t;
					h_lambda[index] = h_lambda[index + 1] = h_lambda[index + 2] = h_lambda[index + 3] = lambda;
					h_lambda[index + 1] = alphas(t);
					index += 4;
				}
			}

	if (fCUDA!=0) {
	
		cuda_copyArrayToDevice(d_vT, h_vT, fN);
		cuda_copyArrayToDevice(d_lambda, h_lambda, fN);

		for(int ic = 0; ic < pts.size(); ++ic) {
			//if (fMode==1) cuda_addCellsBI_PhotonsAMYHigh(d_phi, d_eta, d_vT, d_lambda, d_res, fN, pts[ic], A1, A2, B1, B2, Fq, Nf, aEM, 256, 1);
			//else cuda_addCellsBI_PhotonsAMYLow(d_phi, d_eta, d_vT, d_lambda, d_res, fN, pts[ic], A1, A2, B1, B2, Fq, Nf, aEM, 256, 1);
			cuda_addCellsBI_PhotonsMixAMYHigh(d_phi, d_eta, d_vT, d_lambda, d_res, fN, pts[ic], A1, A2, B1, B2, Fq, Nf, aEM, TcAMY, 256, 1);

			cuda_copyArrayFromDevice(h_res, d_res, 0, fN);

			for(int i=0;i<fN;++i) {
				double t = h_vT[4*i+3];
				if (t<Tcut) continue;
				double sum = 0., sumv1 = 0., sumv2 = 0., sumv3 = 0.;
				sum   = h_res[4*i];
				sumv1 = h_res[4*i+1];
				sumv2 = h_res[4*i+2];
				sumv3 = h_res[4*i+3];
				double tcoef = 2. * f->getDx() * f->getDy() * tau * dtau / 2. / C_PI;
				sum   *= tcoef;
				sumv1 *= tcoef;
				sumv2 *= tcoef;
				sumv3 *= tcoef;
				yield[ic] += sum;
				v1[ic]    += sumv1;
				v2[ic]    += sumv2;
				v3[ic]    += sumv3;
			}
		}
	}
}

double PhotonsMixCUDA::alphas(double T) const {
	return 6. * C_PI / (33. - 2. * Nf) / log(8. * T / TcAMY);
}

double PhotonsMixCUDA::G1(double E, double T) const {
	return A1 * Fq * aEM * alphas(T) * T * T * exp(-E) * log(B1*E/alphas(T)) * pow(gevtofm, 4);
}

double PhotonsMixCUDA::G2(double E, double T) const {
	return A2 * Fq * aEM * alphas(T) * T * T * exp(-E) * log(B2*E/alphas(T)) * pow(gevtofm, 4);
}

double PhotonsMixCUDA::G(double E, double T) const {
	return 1. / C_PI / C_PI * Fq * aEM * alphas(T) * T * T / (exp(E) + 1) * (0.5 * log(3.*E/2./C_PI/alphas(T)) + C12(E) + C34(E)) * pow(gevtofm, 4);
}

double PhotonsMixCUDA::C12(double E) const {
	return (0.041 / E) - 0.3615 + 1.01 * exp(-1.35 * E);
}

double PhotonsMixCUDA::C34(double E) const {
	return sqrt(1. + Nf/6.) * (0.548 / pow(E, 3./2.) * log(12.28 + 1./E) + 0.133 * E / sqrt(1. + E/16.27));
}

double PhotonsMixCUDA::GL(double E, double T,  double lambda, int mode) const {
	//return 4. / 3. / pow(C_PI, 4.) * aEM * 0.3 * T * T * exp(-E) * (2. * log(0.417*E/0.3) + 0.987 * E);
	if (mode==0) return lambda * G1(E,T) + lambda * lambda * (G(E,T) - G1(E,T));
	else return lambda * lambda * G2(E,T) + lambda * (G(E,T) - G2(E,T));
}

double PhotonsMixCUDA::PPR_PKK(double E, double T) {
	if (E>20.) return 0.;
	double RPIK   = 1. / pow(T,3.) * exp(-(5.4018*pow(T,-0.6864)-1.51)*pow(2.*T*E, 0.07) - 0.91*E/T);
	double RPIKs  = pow(T,3.75) * exp(-0.35/pow(2.*T*E,1.05) + (2.3894*pow(T,0.03435) - 3.222)*E/T);
	double RPIKsK = pow(T,3.70) * exp(-(6.096*pow(T,1.889)+1.0299)/pow(2.*T*E,-1.613*pow(T,2.162)+0.975) - 0.96*E/T);
	return RPIK + RPIKs + RPIKsK;
}

double PhotonsMixCUDA::PPR_rho(double E, double T) {
	if (E>20.) return 0.;
	double aT = -31.21 + 353.61*T - 1739.4*T*T + 3105. * T * T * T;
	double bT = -5.513 - 42.2*T + 333.*T*T - 570.*T*T*T;
	double cT = -6.153 + 57.*T - 134.61*T*T + 8.31*T*T*T;
	return exp(aT*E + bT + cT / (E + 0.2));
}

double PhotonsMixCUDA::PPR_brem(double E, double T) {
	if (E>20.) return 0.;
	double abT = -16.28 + 62.45*T - 93.4*T*T - 7.5*T*T*T;
	double bbT = -35.54 + 414.8*T - 2054.*T*T + 3718.8*T*T*T;
	double gbT = 0.7364 - 10.72*T + 56.32*T*T - 103.5*T*T*T;
	double dbT = -2.51 + 58.152*T - 318.24*T*T + 610.7*T*T*T;
	return exp(abT + bbT*E + gbT*E*E + dbT/(E+0.2));
}
