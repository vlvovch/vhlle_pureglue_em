
class EoS;
class Fluid;

// this class takes care of the initial conditions for hydrodynamic evolution
class ICHardSpheres {
  double rho0;  // normalization placeholder (not used)

  double Thickness(double x, double y);  // nuclear thickness profile function
                                           // for optical Glauber approach
  // epsilon: normalization of the initial energy density
  // alpha: parameter relevant to the initial transverse flow
  // b: impact parameter (for optical Glauber)
  double epsilon, b, tau0, alpha, e2;

  // Nucleus parameters
  double A, Ra;
 public:
  ICHardSpheres(double e, double impactPar, double _tau0, double alph_, double A_ = 208, double R_ = 6.5);
  ~ICHardSpheres(void);
  // energy density profile at given point in transverse plane
  double eProfile(double x, double y);
  // Init params
  void init();
  // setIC: initializes entire hydro grid at a given initial proper time tau
  void setIC(Fluid *f, EoS *eos);
  double getrho0() const { return rho0; }
};
