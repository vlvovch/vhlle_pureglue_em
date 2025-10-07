#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include "eos.h"
#include "inc.h"

using namespace std;

const double bagp = pow(247.19 / 197.32, 4) / gevtofm;
const double bagt = pow(247.19 / 197.32, 4) / gevtofm;
const double deg = 16.0 + 3.0 * 12.0 * (7.0 / 8.0);

// EoS choice
//#define TABLE  // Laine, etc
#define SIMPLE  // p=e/3

double EoS::s(double e, double nb, double nq, double ns, double tau) {
  double T, mub, muq, mus, p;
  eos(e, nb, nq, ns, T, mub, muq, mus, p, tau);
  if (T > 0.0)
    return (e + p - mub * nb - muq * nq - mus * ns) / T;
  else
    return 0.;
}

EoSs::EoSs(string fname, int ncols) {
#if defined TABLE || defined LAINE_CFO
  std::vector<double> evec, pvec, Tvec, muvec;
  evec.reserve(10000); pvec.reserve(10000); Tvec.reserve(10000); muvec.reserve(10000);

  ifstream finput(fname.c_str(), ios::in);
  if (!finput) {
    cerr << "can't open input file \"" << fname.c_str() << "\"" << endl;
    exit(1);
  }
  while (true) {
    double e, p, T, mu = 0.0;
    if (ncols == 3) {
      if (!(finput >> e >> p >> T)) break;
    } else {
      if (!(finput >> e >> p >> T >> mu)) break;
    }
    if (p < 0.) p = 0.;
    evec.push_back(e);
    pvec.push_back(p);
    Tvec.push_back(T);
    muvec.push_back(mu);
  }
  finput.close();

  splPE.fill(evec, pvec);
  splTE.fill(evec, Tvec);
  splMU.fill(evec, muvec);

#elif defined SIMPLE
  // nothing
#endif
}

EoSs::~EoSs(void) {}

double EoSs::p(double e) {
#if defined TABLE
  return splPE.f(e);
#elif defined SIMPLE
  return e / 3.;
#endif
}

double EoSs::dpe(double e) {
#if defined TABLE
  return splPE.df(e);
#elif defined SIMPLE
  return 1. / 3.;
#endif
}

double EoSs::t(double e) {
#if defined TABLE
  return splTE.f(e);
#elif defined SIMPLE
  const double cnst =
//      (16 + 0.5 * 21.0 * 2.5) * pow(C_PI, 2) / 30.0 / pow(0.197326968, 3);
	(16 + 0.5 * 21.0 * 3.0) * pow(C_PI, 2) / 30.0 / pow(0.197326968, 3);
//    (16 + 0.5 * 21.0 * 0.0) * pow(C_PI, 2) / 30.0 / pow(0.197326968, 3);
  return e > 0. ? 1.0 * pow(e / cnst, 0.25) : 0.;
#endif
}

double EoSs::mu(double e) {
#if defined TABLE
  return splMU.f(e);
#elif defined SIMPLE
  return 0.;
#endif
}
