#include <fstream>
#include <iomanip>
#include <iomanip>

#include "fld.h"
#include "eos.h"
#include "icGlauber.h"
#include "inc.h"
#include "NumericalIntegration.h"
#include <functional>
#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

// --------------------------------------------
//   Initial state from optical Glauber model
// --------------------------------------------

// Au nucleus parameters for optical Glauber
//const double A = 197.0;    // mass number
//const double Ra = 6.37;    // radius
//const double dlt = 0.54;   // diffuseness
//const double sigma = 4.0;  // NN cross section in fm^2

// LHC PbPb nucleus parameters for optical Glauber
/*const double A = 208.0;    // mass number
const double Ra = 6.6;    // radius
const double dlt = 0.545;   // diffuseness
const double sigma = 7.0;  // NN cross section in fm^2*/

const int nphi = 301;

ICGlauber::ICGlauber(double e, double impactPar, double _tau0, double A_, double R_, double dlt_, double sigma_) {
  epsilon = e;
  b = impactPar;
  tau0 = _tau0;
  A = A_;
  Ra = R_;
  dlt = dlt_;
  sigma = sigma_;
}

ICGlauber::~ICGlauber(void) {}

// Numerical integration helpers (32-point Gauss-Legendre)
static double gauss_legendre_32_1d(double a, double b, const std::function<double(double)>& f) {
  std::vector<double> x, w;
  GetCoefsIntegrateLegendre32(a, b, x, w);
  double sum = 0.;
  for (size_t i = 0; i < x.size(); ++i) sum += w[i] * f(x[i]);
  return sum;
}

double ICGlauber::WoodSaxon1D(double z, double rT, double norm, double R, double a) const {
  return norm / (std::exp((std::sqrt(z*z + rT*rT) - R) / a) + 1.0);
}

double ICGlauber::thickness(double rT) {
  // Integrate along z from -3R to 3R of WoodSaxon1D
  auto f = [this, rT](double z) { return WoodSaxon1D(z, rT, 1.0, Ra, dlt); };
  return gauss_legendre_32_1d(-3.0*Ra, 3.0*Ra, f);
}

double ICGlauber::eProfile(double x, double y) {
  const double rT_p = std::sqrt((x + b / 2.0) * (x + b / 2.0) + y * y);
  const double rT_m = std::sqrt((x - b / 2.0) * (x - b / 2.0) + y * y);
  const double tpp = thickness(rT_p);
  const double tmm = thickness(rT_m);
  return epsilon *
         pow(1. / rho0 * (tpp * (1.0 - pow((1.0 - sigma * tmm / A), A)) +
                          tmm * (1.0 - pow((1.0 - sigma * tpp / A), A))),
             1.0);
}

void ICGlauber::findRPhi(void) {
  _rphi = new double[nphi];
  for (int iphi = 0; iphi < nphi; iphi++) {
    double phi = iphi * C_PI * 2. / (nphi - 1);
    double r = 0., r1 = 0., r2 = 2. * Ra;
    while (fabs((r2 - r1) / r2) > 0.001 && r2 > 0.001) {
      r = 0.5 * (r1 + r2);
      if (eProfile(r * cos(phi), r * sin(phi)) > 0.5)
        r1 = r;
      else
        r2 = r;
    }
    _rphi[iphi] = r;
  }
}


double ICGlauber::rPhi(double phi) {
  const double cpi = C_PI;
  phi = phi - 2. * cpi * floor(phi / 2. / cpi);
  int iphi = (int)(phi / (2. * cpi) * (nphi - 1));
  int iphi1 = iphi + 1;
  if (iphi1 == nphi) iphi = nphi - 2;
  return _rphi[iphi] * (1. - (phi / (2. * cpi) * (nphi - 1) - iphi)) +
         _rphi[iphi1] * (phi / (2. * cpi) * (nphi - 1) - iphi);
}

void ICGlauber::init() {
  // Compute normalization constant intgr2 = ∬ thickness(x,y) dx dy over square [-3R,3R]^2
  std::vector<double> xleg, wleg;
  GetCoefsIntegrateLegendre32(-3.0*Ra, 3.0*Ra, xleg, wleg);
  auto integrand = [this](double X, double Y){
    double rT = std::sqrt(X*X + Y*Y);
    return thickness(rT);
  };
  // Tensor product quadrature
  double sum = 0.0;
  for (size_t i=0;i<xleg.size();++i){
    double xi = xleg[i];
    double wi = wleg[i];
    for (size_t j=0;j<xleg.size();++j){
      double yj = xleg[j];
      double wj = wleg[j];
      sum += wi * wj * integrand(xi,yj);
    }
  }
  const double intgr2 = sum;
  if (intgr2 == 0.0) {
    cerr << "IC::init Error! integral == 0\n";
    exit(1);
  }
  cout << "a = " << A / intgr2 << endl;
  // Evaluate thickness at rT=0 to get rho0
  const double tpp = thickness(0.0);
  rho0 = 2.0 * tpp * (1.0 - pow((1.0 - sigma * tpp / A), A));
}

void ICGlauber::setIC(Fluid *f, EoS *eos) {
  double e, nb, nq, vx = 0., vy = 0., vz = 0.;
  Cell *c;
  ofstream fvel("velocity_debug.txt");

  init();

  findRPhi();  // fill in R(phi) table
  cout << "R(phi) =  ";
  for (int jj = 0; jj < 5; jj++) cout << rPhi(jj * C_PI / 2.) << "  ";  // test
  cout << endl;
  //--------------
  double avv_num = 0., avv_den = 0.;
  double Etotal = 0.0;

  for (int ix = 0; ix < f->getNX(); ix++)
    for (int iy = 0; iy < f->getNY(); iy++)
      for (int iz = 0; iz < f->getNZ(); iz++) {
        c = f->getCell(ix, iy, iz);
        double x = f->getX(ix);
        double y = f->getY(iy);
        double eta = f->getZ(iz);
        double etaFactor;
        //double eta1 = fabs(eta) < 1.3 ? 0.0 : fabs(eta) - 1.3;
        //etaFactor = exp(-eta1 * eta1 / 2.1 / 2.1) * (fabs(eta) < 5.3 ? 1.0 : 0.0);
		if (fabs(eta)<3.0) etaFactor = 1.;
		else etaFactor = exp(-(fabs(eta)-3.0)*(fabs(eta)-3.0)/2./0.4/0.4);
        e = eProfile(x, y) * etaFactor;
        //if (e < 0.5) e = 0.0;
        vx = vy = 0.0;
		nb = nq = 0.0;
        //nb = nq = eProfile(x, y) / 0.5;
        vz = 0.0;

      avv_num += sqrt(vx * vx + vy * vy) * e;
      avv_den += e;

        c->setPrimVar(eos, tau0, e, nb, nq, 0., vx, vy, vz);
        double _p = eos->p(e, nb, nq, 0., c->getTauP());
        const double gamma2 = 1.0 / (1.0 - vx * vx - vy * vy - vz * vz);
        Etotal +=
            ((e + _p) * gamma2 * (cosh(eta) + vz * sinh(eta)) - _p * cosh(eta));
        c->saveQprev();

        if (e > 0.) c->setAllM(1.);
      }
  fvel.close();
  cout << "average initial flow = " << avv_num / avv_den << endl;
  cout << "total energy = " << Etotal *f->getDx() * f->getDy() * f->getDz() *
                                   tau0 << endl;
}

// old ROOT-based integrators removed
