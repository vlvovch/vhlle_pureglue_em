#include "photons.h"
#include "inc.h"
#include "NumericalIntegration.h"
#include "fld.h"
#include "eos.h"
#include "cll.h"

Photons::Photons(double Tcut_) : Tcut(Tcut_)
{
}

Photons::Photons(const std::vector<double> & ptin, const std::vector<double> & yin, double Tcut_) : pts(ptin), ys(yin), Tcut(Tcut_)
{
	yield.resize(ptin.size());
	v1.resize(ptin.size());
	v2.resize(ptin.size());
	v3.resize(ptin.size());
}


Photons::~Photons()
{
}

void Photons::addPtY(double pt, double y) { 
	pts.push_back(pt); 
	ys.push_back(y); 
	yield.push_back(0.); 
	v1.push_back(0.); 
	v2.push_back(0.); 
	v3.push_back(0.); 
}

PhotonSpectrum Photons::GetSpectrum() const {
	PhotonSpectrum ret;
	for(int ic=0;ic<pts.size();++ic) {
		PhotonSpectrumEntry ent;
		ent.pt    = pts[ic];
		ent.y     = ys[ic];
		ent.yield = yield[ic];
		ent.v1    = v1[ic] / yield[ic];
		ent.v2    = v2[ic] / yield[ic];
		ent.v3    = v3[ic] / yield[ic];
		ret.Entries.push_back(ent);
	}
	return ret;
}

PhotonsQGP::PhotonsQGP(const std::vector<double> & ptin, const std::vector<double> & yin, double tau0_, double taus_, double Tcut_) : Photons(ptin, yin, Tcut_), tau0(tau0_), taus(taus_)
{
}

void PhotonsQGP::init() {
	coef = 7.297352e-3 * 0.3 / pow(C_PI, 3.) / 4. * (2. / 3.);
	coefBI = coef * 2.;
	//a = 0.197; b = 0.987; cc = 0.884;
	a = 0.232; b = 0.987; cc = 0.884;
	alphas = 0.3;

	GetCoefsIntegrateLegendre32(0., 2*C_PI, xphi, wphi);
	GetCoefsIntegrateLaguerre32(xeta, weta);
}

void PhotonsQGP::addCell(double tau, double dtau, Fluid *f, EoS *eos, Cell *c) {
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
}

void PhotonsQGP::addCellBI(double tau, double dtau, Fluid *f, EoS *eos, Cell *c) {
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

void PhotonsQGP::addCellBISymm(double tau, double dtau, Fluid *f, EoS *eos, Cell *c) {
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

PhotonsAMY::PhotonsAMY(const std::vector<double> & ptin, const std::vector<double> & yin, double tau0_, double taus_, double Tcut_) : Photons(ptin, yin, Tcut_), tau0(tau0_), taus(taus_)
{
}

void PhotonsAMY::init() {
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

void PhotonsAMY::addCell(double tau, double dtau, Fluid *f, EoS *eos, Cell *c) {
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
		double tcoef = f->getDx() * f->getDy() * f->getDz() * tau * dtau / 2. / C_PI;
		sum       *= tcoef;
		sumv1     *= tcoef;
		sumv2     *= tcoef;
		sumv3     *= tcoef;
		yield[ic] += sum;
		v1[ic]    += sumv1;
		v2[ic]    += sumv2;
		v3[ic]    += sumv3;
	}
}

void PhotonsAMY::addCellBI(double tau, double dtau, Fluid *f, EoS *eos, Cell *c) {
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

void PhotonsAMY::addCellBISymm(double tau, double dtau, Fluid *f, EoS *eos, Cell *c) {
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

double PhotonsAMY::alphas(double T) const {
	return 6. * C_PI / (33. - 2. * Nf) / log(8. * T / Tc);
}

double PhotonsAMY::G1(double E, double T) const {
	return A1 * Fq * aEM * alphas(T) * T * T * exp(-E) * log(B1*E/alphas(T)) * pow(gevtofm, 4);
}

double PhotonsAMY::G2(double E, double T) const {
	return A2 * Fq * aEM * alphas(T) * T * T * exp(-E) * log(B2*E/alphas(T)) * pow(gevtofm, 4);
}

double PhotonsAMY::G(double E, double T) const {
	return 1. / C_PI / C_PI * Fq * aEM * alphas(T) * T * T / (exp(E) + 1) * (0.5 * log(3.*E/2./C_PI/alphas(T)) + C12(E) + C34(E)) * pow(gevtofm, 4);
}

double PhotonsAMY::C12(double E) const {
	return (0.041 / E) - 0.3615 + 1.01 * exp(-1.35 * E);
}

double PhotonsAMY::C34(double E) const {
	return sqrt(1. + Nf/6.) * (0.548 / pow(E, 3./2.) * log(12.28 + 1./E) + 0.133 * E / sqrt(1. + E/16.27));
}

double PhotonsAMY::GL(double E, double T,  double lambda, int mode) const {
	//return 4. / 3. / pow(C_PI, 4.) * aEM * 0.3 * T * T * exp(-E) * (2. * log(0.417*E/0.3) + 0.987 * E);
	if (mode==0) return lambda * G1(E,T) + lambda * lambda * (G(E,T) - G1(E,T));
	else return lambda * lambda * G2(E,T) + lambda * (G(E,T) - G2(E,T));
}


// HADRONS

PhotonsHadrons::PhotonsHadrons(const std::vector<double> & ptin, const std::vector<double> & yin, double tau0_, double taus_, double Tcut_) : Photons(ptin, yin, Tcut_), tau0(tau0_), taus(taus_)
{
}

void PhotonsHadrons::init() {
	GetCoefsIntegrateLegendre32(0., 2.*C_PI, xphi, wphi);
	GetCoefsIntegrateLaguerre32(xeta, weta);
}

PhotonsHadrons::~PhotonsHadrons() {
}

void PhotonsHadrons::addCellBI(double tau, double dtau, Fluid *f, EoS *eos, Cell *c) {
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

void PhotonsHadrons::addCellBISymm(double tau, double dtau, Fluid *f, EoS *eos, Cell *c) {
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



double PhotonsHadrons::PPR_PKK(double E, double T) {
	double RPIK   = 1. / pow(T,3.) * exp(-(5.4018*pow(T,-0.6864)-1.51)*pow(2.*T*E, 0.07) - 0.91*E/T);
	double RPIKs  = pow(T,3.75) * exp(-0.35/pow(2.*T*E,1.05) + (2.3894*pow(T,0.03435) - 3.222)*E/T);
	double RPIKsK = pow(T,3.70) * exp(-(6.096*pow(T,1.889)+1.0299)/pow(2.*T*E,-1.613*pow(T,2.162)+0.975) - 0.96*E/T);
	return RPIK + RPIKs + RPIKsK;
}

double PhotonsHadrons::PPR_rho(double E, double T) {
	double aT = -31.21 + 353.61*T - 1739.4*T*T + 3105. * T * T * T;
	double bT = -5.513 - 42.2*T + 333.*T*T - 570.*T*T*T;
	double cT = -6.153 + 57.*T - 134.61*T*T + 8.31*T*T*T;
	return exp(aT*E + bT + cT / (E + 0.2));
}

double PhotonsHadrons::PPR_brem(double E, double T) {
	double abT = -16.28 + 62.45*T - 93.4*T*T - 7.5*T*T*T;
	double bbT = -35.54 + 414.8*T - 2054.*T*T + 3718.8*T*T*T;
	double gbT = 0.7364 - 10.72*T + 56.32*T*T - 103.5*T*T*T;
	double dbT = -2.51 + 58.152*T - 318.24*T*T + 610.7*T*T*T;
	return exp(abT + bbT*E + gbT*E*E + dbT/(E+0.2));
}


// MIX

PhotonsMix::PhotonsMix(const std::vector<double> & ptin, const std::vector<double> & yin, double tau0_, double taus_, double Tcut_, double Tc_) : Photons(ptin, yin, Tcut_), TcAMY(Tc_), tau0(tau0_), taus(taus_)
{
}

void PhotonsMix::init() {
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

void PhotonsMix::addCell(double tau, double dtau, Fluid *f, EoS *eos, Cell *c) {
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
			if (t>TcAMY) sumphi += GL(etil/t, t, lambda, fMode);
			else sumphi += PPR(etil, t);
			sum    += wphi[iphi]                * sumphi;
			sumv1  += wphi[iphi] * cos(tphi)    * sumphi;
			sumv2  += wphi[iphi] * cos(2.*tphi) * sumphi;
			sumv3  += wphi[iphi] * cos(3.*tphi) * sumphi;
		}

		//double tcoef = coef * f->getDx() * f->getDy() * f->getDz() * tau * dtau * t * t * pow(gevtofm, 4);
		double tcoef = f->getDx() * f->getDy() * f->getDz() * tau * dtau / 2. / C_PI;
		sum       *= tcoef;
		sumv1     *= tcoef;
		sumv2     *= tcoef;
		sumv3     *= tcoef;
		yield[ic] += sum;
		v1[ic]    += sumv1;
		v2[ic]    += sumv2;
		v3[ic]    += sumv3;
	}
}

void PhotonsMix::addCellBI(double tau, double dtau, Fluid *f, EoS *eos, Cell *c) {
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
				if (t>TcAMY) sumphi += weta[ieta] * GL(etil/t, t, lambda, fMode);
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
	}
}

void PhotonsMix::addCellBISymm(double tau, double dtau, Fluid *f, EoS *eos, Cell *c) {
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

double PhotonsMix::alphas(double T) const {
	return 6. * C_PI / (33. - 2. * Nf) / log(8. * T / Tc);
}

double PhotonsMix::G1(double E, double T) const {
	return A1 * Fq * aEM * alphas(T) * T * T * exp(-E) * log(B1*E/alphas(T)) * pow(gevtofm, 4);
}

double PhotonsMix::G2(double E, double T) const {
	return A2 * Fq * aEM * alphas(T) * T * T * exp(-E) * log(B2*E/alphas(T)) * pow(gevtofm, 4);
}

double PhotonsMix::G(double E, double T) const {
	return 1. / C_PI / C_PI * Fq * aEM * alphas(T) * T * T / (exp(E) + 1) * (0.5 * log(3.*E/2./C_PI/alphas(T)) + C12(E) + C34(E)) * pow(gevtofm, 4);
}

double PhotonsMix::C12(double E) const {
	return (0.041 / E) - 0.3615 + 1.01 * exp(-1.35 * E);
}

double PhotonsMix::C34(double E) const {
	return sqrt(1. + Nf/6.) * (0.548 / pow(E, 3./2.) * log(12.28 + 1./E) + 0.133 * E / sqrt(1. + E/16.27));
}

double PhotonsMix::GL(double E, double T,  double lambda, int mode) const {
	//return 4. / 3. / pow(C_PI, 4.) * aEM * 0.3 * T * T * exp(-E) * (2. * log(0.417*E/0.3) + 0.987 * E);
	if (mode==0) return lambda * G1(E,T) + lambda * lambda * (G(E,T) - G1(E,T));
	else return lambda * lambda * G2(E,T) + lambda * (G(E,T) - G2(E,T));
}

double PhotonsMix::PPR_PKK(double E, double T) {
	double RPIK   = 1. / pow(T,3.) * exp(-(5.4018*pow(T,-0.6864)-1.51)*pow(2.*T*E, 0.07) - 0.91*E/T);
	double RPIKs  = pow(T,3.75) * exp(-0.35/pow(2.*T*E,1.05) + (2.3894*pow(T,0.03435) - 3.222)*E/T);
	double RPIKsK = pow(T,3.70) * exp(-(6.096*pow(T,1.889)+1.0299)/pow(2.*T*E,-1.613*pow(T,2.162)+0.975) - 0.96*E/T);
	return RPIK + RPIKs + RPIKsK;
}

double PhotonsMix::PPR_rho(double E, double T) {
	double aT = -31.21 + 353.61*T - 1739.4*T*T + 3105. * T * T * T;
	double bT = -5.513 - 42.2*T + 333.*T*T - 570.*T*T*T;
	double cT = -6.153 + 57.*T - 134.61*T*T + 8.31*T*T*T;
	return exp(aT*E + bT + cT / (E + 0.2));
}

double PhotonsMix::PPR_brem(double E, double T) {
	double abT = -16.28 + 62.45*T - 93.4*T*T - 7.5*T*T*T;
	double bbT = -35.54 + 414.8*T - 2054.*T*T + 3718.8*T*T*T;
	double gbT = 0.7364 - 10.72*T + 56.32*T*T - 103.5*T*T*T;
	double dbT = -2.51 + 58.152*T - 318.24*T*T + 610.7*T*T*T;
	return exp(abT + bbT*E + gbT*E*E + dbT/(E+0.2));
}
