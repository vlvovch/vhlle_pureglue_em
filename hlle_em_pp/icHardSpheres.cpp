#include <fstream>
#include <iomanip>
#include <iomanip>
#include <cmath>
#include <iostream>

#include "fld.h"
#include "eos.h"
#include "icHardSpheres.h"
#include "inc.h"

using namespace std;


ICHardSpheres::ICHardSpheres(double e, double impactPar, double _tau0, double alph_, double A_, double R_) {
  epsilon = e;
  b = impactPar;
  tau0 = _tau0;
  A = A_;
  Ra = R_;
  alpha = alph_;
  e2 = 1.;
}

ICHardSpheres::~ICHardSpheres(void) {}

double ICHardSpheres::eProfile(double x, double y) {
	return e2 * Thickness(x + b/ 2.0, y) * Thickness(x - b/2.0, y);
}

void ICHardSpheres::init() {
	e2 = epsilon / Thickness(0.,0.) / Thickness(0.,0.); 
}

void ICHardSpheres::setIC(Fluid *f, EoS *eos) {
  double e, nb, nq, vx = 0., vy = 0., vz = 0.;
  Cell *c;
  ofstream fvel("velocity_debug.txt");

  init();

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
		double etafl = 2.95, etagauss = 0.5;
		if (fabs(eta)<etafl) etaFactor = 1.;
		else etaFactor = exp(-(fabs(eta)-etafl)*(fabs(eta)-etafl)/2./etagauss/etagauss);
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

		//cout << x << " " << y << " " << e << "\n";

        if (e > 0.) c->setAllM(1.);
      }
  fvel.close();
  cout << "average initial flow = " << avv_num / avv_den << endl;
  cout << "total energy = " << Etotal * f->getDx() * f->getDy() * f->getDz() * tau0 << endl;
}

double ICHardSpheres::Thickness(double x, double y) {
  double coef = 3. / 4. / C_PI / Ra / Ra / Ra;
  if (x*x + y*y < Ra*Ra) return 2. * coef * sqrt(Ra*Ra - x*x - y*y);
  else return 0.;
}
