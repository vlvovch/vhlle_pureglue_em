#ifndef PHOTONSCUDA_H
#define PHOTONSCUDA_H

#include "photons.h"

class PhotonsQGPCUDA : public Photons
{
	double coef;
	double coefBI;
	double a, b, cc;
	double alphas;
	double tau0, taus;                // Undersaturated QGP
	std::vector<double> xphi, wphi;   // Legendre quadrature to integrate over phi
	std::vector<double> xeta, weta;   // Laguerre quadrature to integrate over eta (for boost-invariant case)

	int fN;

	float *h_vT;
  float *h_lambda;
	float *d_vT;
  float *d_lambda;
	float *h_phi;
	float *d_phi;
	float *h_eta;
	float *d_eta;
	float *h_res;
	float *d_res;
	int fCUDA;
public:
	PhotonsQGPCUDA(double tau0_=0.1, double taus_=0., double Tcut_ = 0.155) : Photons(Tcut_), tau0(tau0_), taus(taus_) { fN = 0; fCUDA = 0; init(); }
	PhotonsQGPCUDA(const std::vector<double> & ptin, const std::vector<double> & yin, double tau0_=0.1, double taus_=0., double Tcut_ = 0.155);
	void init();
	void initCUDAArrays(Fluid *f);
	virtual ~PhotonsQGPCUDA();// { }
	//virtual void addCell(double tau, double dtau, Fluid *f, EoS *eos, Cell *c);
	virtual void addCell(double tau, double dtau, Fluid *f, EoS *eos, Cell *c) { }
	virtual void addCellBI(double tau, double dtau, Fluid *f, EoS *eos, Cell *c);
	virtual void addCellBISymm(double tau, double dtau, Fluid *f, EoS *eos, Cell *c);
	virtual void addCellsBI(double tau, double dtau, Fluid *f, EoS *eos);
	virtual void addCellsBISymm(double tau, double dtau, Fluid *f, EoS *eos);
	void setMode(int mode_) { }
};


class PhotonsAMYCUDA : public Photons
{
	double coef;
	double coefBI;
	double A1, A2;
	double B1, B2;
	double Fq;
	double Nf;
	double Tc;
	//double alphas;
	double aEM;
	double tau0, taus;                // Undersaturated QGP
	std::vector<double> xphi, wphi;   // Legendre quadrature to integrate over phi
	std::vector<double> xeta, weta;   // Laguerre quadrature to integrate over eta (for boost-invariant case)
	int fMode;

	int fN;
public:
	float *h_vT;
    float *h_lambda;
	float *d_vT;
    float *d_lambda;
	float *h_phi;
	float *d_phi;
	float *h_eta;
	float *d_eta;
	float *h_res;
	float *d_res;

	int fCUDA;
public:
	PhotonsAMYCUDA(double tau0_=0.1, double taus_=0., double Tcut_ = 0.155) : Photons(Tcut_), tau0(tau0_), taus(taus_), fMode(0) { fN = 0; fMode = 0; init(); }
	PhotonsAMYCUDA(const std::vector<double> & ptin, const std::vector<double> & yin, double tau0_=0.1, double taus_=0., double Tcut_ = 0.155);
	void init();
	void initCUDAArrays(Fluid *f);
	void setMode(int mode_ = 0) { fMode = mode_; }
	virtual ~PhotonsAMYCUDA();// { }
	virtual void addCell(double tau, double dtau, Fluid *f, EoS *eos, Cell *c) { }
	virtual void addCellBI(double tau, double dtau, Fluid *f, EoS *eos, Cell *c);
	virtual void addCellBISymm(double tau, double dtau, Fluid *f, EoS *eos, Cell *c);
	virtual void addCellsBI(double tau, double dtau, Fluid *f, EoS *eos);
	virtual void addCellsBISymm(double tau, double dtau, Fluid *f, EoS *eos);
	double alphas(double T) const;
	double C12(double E) const;
	double C34(double E) const;
	double G1(double E, double T) const;
	double G2(double E, double T) const;
	double G(double E, double T) const;
	double GL(double E, double T, double la, int mode = 0) const;
	double GetPrefactor(double T) const { return aEM * alphas(T) * Fq * T * T / 3.1415926535 / 3.1415926535; }
};


class PhotonsHadronsCUDA : public Photons
{
	double tau0, taus;                // Undersaturated QGP
	std::vector<double> xphi, wphi;   // Legendre quadrature to integrate over phi
	std::vector<double> xeta, weta;   // Laguerre quadrature to integrate over eta (for boost-invariant case)
	int fMode;

	int fN;
public:
	float *h_vT;
    float *h_lambda;
	float *d_vT;
    float *d_lambda;
	float *h_phi;
	float *d_phi;
	float *h_eta;
	float *d_eta;
	float *h_res;
	float *d_res;

	int fCUDA;
public:
	PhotonsHadronsCUDA(double tau0_=0.1, double taus_=0., double Tcut_ = 0.155) : Photons(Tcut_), tau0(tau0_), taus(taus_), fMode(0) { fN = 0; fMode = 0; init(); }
	PhotonsHadronsCUDA(const std::vector<double> & ptin, const std::vector<double> & yin, double tau0_=0.1, double taus_=0., double Tcut_ = 0.155);
	void init();
	void initCUDAArrays(Fluid *f);
	void setMode(int mode_ = 0) { fMode = mode_; }
	virtual ~PhotonsHadronsCUDA();// { }
	virtual void addCell(double tau, double dtau, Fluid *f, EoS *eos, Cell *c) { }
	virtual void addCellBI(double tau, double dtau, Fluid *f, EoS *eos, Cell *c);
	virtual void addCellBISymm(double tau, double dtau, Fluid *f, EoS *eos, Cell *c);
	virtual void addCellsBI(double tau, double dtau, Fluid *f, EoS *eos);
	virtual void addCellsBISymm(double tau, double dtau, Fluid *f, EoS *eos);
	double PPR_PKK(double E, double T);
	double PPR_rho(double E, double T);
	double PPR_brem(double E, double T);
	double PPR(double E, double T) { 
		return PPR_PKK(E,T) + PPR_rho(E,T) + PPR_brem(E,T); 
	}
};

class PhotonsMixCUDA : public Photons
{
	double coef;
	double coefBI;
	double A1, A2;
	double B1, B2;
	double Fq;
	double Nf;
	double TcAMY;
	double Tsw;
	//double alphas;
	double aEM;
	double tau0, taus;                // Undersaturated QGP
	std::vector<double> xphi, wphi;   // Legendre quadrature to integrate over phi
	std::vector<double> xeta, weta;   // Laguerre quadrature to integrate over eta (for boost-invariant case)
	int fMode;

	int fN;
public:
	float *h_vT;
  float *h_lambda;
	float *d_vT;
  float *d_lambda;
	float *h_phi;
	float *d_phi;
	float *h_eta;
	float *d_eta;
	float *h_res;
	float *d_res;

	int fCUDA;
public:
	PhotonsMixCUDA(double tau0_=0.1, double taus_=0., double Tcut_ = 0.155, double Tc_ = 0.155) : Photons(Tcut_), tau0(tau0_), taus(taus_), Tsw(Tc_), fMode(0) { fN = 0; fMode = 0; init(); }
	PhotonsMixCUDA(const std::vector<double> & ptin, const std::vector<double> & yin, double tau0_=0.1, double taus_=0., double Tcut_ = 0.155, double Tc_ = 0.155);
	void init();
	void initCUDAArrays(Fluid *f);
	void setMode(int mode_ = 0) { fMode = mode_; }
	virtual ~PhotonsMixCUDA();// { }
	virtual void addCell(double tau, double dtau, Fluid *f, EoS *eos, Cell *c) { }
	virtual void addCellBI(double tau, double dtau, Fluid *f, EoS *eos, Cell *c);
	virtual void addCellBISymm(double tau, double dtau, Fluid *f, EoS *eos, Cell *c);
	virtual void addCellsBI(double tau, double dtau, Fluid *f, EoS *eos);
	virtual void addCellsBISymm(double tau, double dtau, Fluid *f, EoS *eos) { }
	double alphas(double T) const;

	double C12(double E) const;
	double C34(double E) const;
	double G1(double E, double T) const;
	double G2(double E, double T) const;
	double G(double E, double T) const;
	double GL(double E, double T, double la, int mode = 0) const;
	double GetPrefactor(double T) const { return aEM * alphas(T) * Fq * T * T / 3.1415926535 / 3.1415926535; }

	double PPR_PKK(double E, double T);
	double PPR_rho(double E, double T);
	double PPR_brem(double E, double T);
	double PPR(double E, double T) { 
		return PPR_PKK(E,T) + PPR_rho(E,T) + PPR_brem(E,T); 
	}
};

#endif
