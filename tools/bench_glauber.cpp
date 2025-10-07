// Simple benchmark to validate Gauss-Legendre integration used in ICGlauber
// Compares 32-point vs 40-point quadrature for thickness and 2D normalization

#include <cmath>
#include <iostream>
#include <vector>
#include "NumericalIntegration.h"

static double wood_saxon_1d(double z, double rT, double norm, double R, double a) {
  return norm / (std::exp((std::sqrt(z*z + rT*rT) - R) / a) + 1.0);
}

static double thickness_GL(double rT, double R, double a, int order) {
  std::vector<double> x, w;
  if (order == 32) GetCoefsIntegrateLegendre32(-3.0*R, 3.0*R, x, w);
  else if (order == 40) GetCoefsIntegrateLegendre40(-3.0*R, 3.0*R, x, w);
  else if (order == 10) GetCoefsIntegrateLegendre10(-3.0*R, 3.0*R, x, w);
  else GetCoefsIntegrateLegendre32(-3.0*R, 3.0*R, x, w);
  double sum = 0.0;
  for (size_t i=0;i<x.size();++i) sum += w[i] * wood_saxon_1d(x[i], rT, 1.0, R, a);
  return sum;
}

static double normalization_2D_GL(double R, double a, int order) {
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

int main() {
  const double Ra = 6.6;   // fm
  const double dlt = 0.545; // fm
  const double A = 208.0;

  std::cout << "Gauss-Legendre benchmark (Pb, R=6.6, a=0.545)\n";
  for (double rT : {0.0, 2.0, 4.0, 6.0, 8.0}) {
    double t32 = thickness_GL(rT, Ra, dlt, 32);
    double t40 = thickness_GL(rT, Ra, dlt, 40);
    std::cout << "rT=" << rT << "  T32=" << t32 << "  T40=" << t40
              << "  rel.diff=" << (t40 - t32) / (std::abs(t40) + 1e-16) << "\n";
  }

  double I32 = normalization_2D_GL(Ra, dlt, 32);
  double I40 = normalization_2D_GL(Ra, dlt, 40);
  std::cout << "2D integral: I32=" << I32 << "  I40=" << I40
            << "  rel.diff=" << (I40 - I32) / (std::abs(I40) + 1e-16) << "\n";
  std::cout << "a = A / I32 = " << A / I32 << ",  A / I40 = " << A / I40 << "\n";
  return 0;
}
