
class EoS;
class Fluid;

// this class takes care of the initial conditions for hydrodynamic evolution
class ICGlauber {
  double rho0;  // normalization factor for participant density
  double *_rphi;
  void findRPhi(void);
  double rPhi(double phi);
  // Numerical helpers
  double WoodSaxon1D(double z, double rT, double norm, double R, double a) const;
  double thickness(double rT);
  // epsilon: normalization of the initial energy density
  // alpha: parameter relevant to the initial transverse flow
  // b: impact parameter (for optical Glauber)
  double epsilon, b, tau0;

  // Nucleus parameters
  double A, Ra, dlt, sigma;
 public:
  ICGlauber(double e, double impactPar, double _tau0, double A_ = 208, double R_ = 6.5, double dlt_ = 0.54, double sigma_ = 4.);
  ~ICGlauber(void);
  // energy density profile at given point in transverse plane
  double eProfile(double x, double y);
  // Init params
  void init();
  // setIC: initializes entire hydro grid at a given initial proper time tau
  void setIC(Fluid *f, EoS *eos);
  double getrho0() const { return rho0; }
};
