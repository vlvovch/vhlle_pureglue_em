#include <cmath>
#include <iostream>
#include <vector>

#include <TF1.h>
#include <TF2.h>

#include "NumericalIntegration.h"

// ROOT TF1-compatible function for Woods-Saxon along z at fixed rT
static double ws_profile(double* x, double* p) {
  double z = x[0];
  double rT = p[0];
  double R  = p[1];
  double a  = p[2];
  double norm = 1.0;
  return norm / (std::exp((std::sqrt(z*z + rT*rT) - R) / a) + 1.0);
}

static double thickness_GL(double rT, double R, double a, int order) {
  std::vector<double> x, w;
  if (order == 32) GetCoefsIntegrateLegendre32(-3.0*R, 3.0*R, x, w);
  else if (order == 40) GetCoefsIntegrateLegendre40(-3.0*R, 3.0*R, x, w);
  else if (order == 10) GetCoefsIntegrateLegendre10(-3.0*R, 3.0*R, x, w);
  else GetCoefsIntegrateLegendre32(-3.0*R, 3.0*R, x, w);
  double sum = 0.0;
  for (size_t i=0;i<x.size();++i) {
    double params[3] = {rT, R, a};
    sum += w[i] * ws_profile(&x[i], params);
  }
  return sum;
}

static double thickness_TF1(double rT, double R, double a) {
  TF1 fz("ws_z", ws_profile, -3.0*R, 3.0*R, 3);
  fz.SetParameters(rT, R, a);
  return fz.Integral(-3.0*R, 3.0*R, 1e-9);
}

static double normalization_GL(double R, double a, int order) {
  std::vector<double> x, w;
  if (order == 32) GetCoefsIntegrateLegendre32(-3.0*R, 3.0*R, x, w);
  else if (order == 40) GetCoefsIntegrateLegendre40(-3.0*R, 3.0*R, x, w);
  else if (order == 10) GetCoefsIntegrateLegendre10(-3.0*R, 3.0*R, x, w);
  else GetCoefsIntegrateLegendre32(-3.0*R, 3.0*R, x, w);
  double sum = 0.0;
  for (size_t i=0;i<x.size();++i){
    double xi = x[i];
    double wi = w[i];
    for (size_t j=0;j<x.size();++j){
      double yj = x[j];
      double wj = w[j];
      double rT = std::sqrt(xi*xi + yj*yj);
      sum += wi * wj * thickness_GL(rT, R, a, order);
    }
  }
  return sum;
}

// TF2 integrand that internally uses TF1 to integrate along z for each (x,y)
static double thickness_2D_integrand(double* xy, double* p) {
  double X = xy[0], Y = xy[1];
  double R = p[0], a = p[1];
  double rT = std::sqrt(X*X + Y*Y);
  return thickness_TF1(rT, R, a);
}

static double normalization_TF2(double R, double a) {
  TF2 fxy("thickness2D", thickness_2D_integrand, -3.0*R, 3.0*R, -3.0*R, 3.0*R, 2);
  fxy.SetParameters(R, a);
  return fxy.Integral(-3.0*R, 3.0*R, -3.0*R, 3.0*R, 1e-9);
}

int main() {
  const double Ra = 6.6;   // fm
  const double dlt = 0.545; // fm
  const double A = 208.0;
  std::cout << "ROOT vs GL benchmark (Pb, R=6.6, a=0.545)\n";
  for (double rT : {0.0, 2.0, 4.0, 6.0, 8.0}) {
    double tTF = thickness_TF1(rT, Ra, dlt);
    double t32 = thickness_GL(rT, Ra, dlt, 32);
    std::cout << "rT=" << rT << "  TF1=" << tTF << "  GL32=" << t32
              << "  rel.diff=" << (t32 - tTF) / (std::abs(tTF) + 1e-16) << "\n";
  }
  double ITF = normalization_TF2(Ra, dlt);
  double I32 = normalization_GL(Ra, dlt, 32);
  std::cout << "2D integral: TF2=" << ITF << "  GL32=" << I32
            << "  rel.diff=" << (I32 - ITF) / (std::abs(ITF) + 1e-16) << "\n";
  std::cout << "a = A / TF2 = " << A / ITF << ",  A / GL32 = " << A / I32 << "\n";
  return 0;
}
