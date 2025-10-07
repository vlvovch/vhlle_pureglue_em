#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <ctime>
#include <sstream>
#include "inc.h"
#include "eos.h"
#include "eoSimpleSpline.h"
#include "eosLinearCombination.h"

#ifdef _DEBUG
#pragma comment(lib,"libHLLEviscD.lib")
#else
#pragma comment(lib,"libHLLEvisc.lib")
#endif

using namespace std;

int main(int argc, char **argv) {
  // pointers to all the main objects
  time_t start = 0, end;

  time(&start);

  EoSimpleSpline *eos1 = new EoSimpleSpline("eos/Lattice_BW_QGP.dat");
  EoSimpleSpline *eos2 = new EoSimpleSpline("eos/Lattice_BW_YM.dat");

  fstream fout("Lattice-QCD-YM-EoS.dat", fstream::out);

  fout.close();

	{
		fout.open("Lattice-QCD-Full-YM-qqbar.dat", fstream::out);

		double Tmin = 0.001, Tmax = 1.000001;
		double dT = 0.001;

		for(double T=Tmin;T<=Tmax;T+=dT) {
			fout << setw(15) << T;
			double ten1 = eos1->splET.f(T);
			double ten2 = eos2->splET.f(T);
			fout << setw(15) << eos1->p(ten1) / pow(T, 4) / pow(gevtofm, 3);
			fout << setw(15) << eos2->p(ten2) / pow(T, 4) / pow(gevtofm, 3);
			fout << setw(15) << ( eos1->p(ten1) - eos2->p(ten2)) / pow(T, 4) / pow(gevtofm, 3);
			fout << endl;
		}

		fout.close();
	}

  fout.open("Lattice-QCD-YM-EoS-lheat.dat", fstream::out);
  {
	  fout << setw(15) << "lambda"
	   << setw(15) << "de/T4"
	   << setw(15) << "ds/T3"
	   << setw(15) << "de/e"
	   << setw(15) << "ds/s"
	   << endl;

	  double lmin = 0., lmax = 1.001;
	  double dl = 0.01;
	  //double dT = 0.0001;

	  for(double la=lmin;la<=lmax;la+=dl) {
		EoSLinearCombination eos(eos1, eos2, 0., 0.);
		fout << setw(15) << la;
		fout << setw(15) << (eos.en(la, 0.2700001) - eos.en(la, 0.2699999))/0.270/0.270/0.270/0.270/gevtofm/gevtofm/gevtofm;
		fout << setw(15) << (eos.entropy(la, 0.2700001)  - eos.entropy(la, 0.2699999))/0.270/0.270/0.270/gevtofm/gevtofm/gevtofm;
		fout << setw(15) << (eos.en(la, 0.2700001) - eos.en(la, 0.2699999))/eos.en(la, 0.2700001);
		fout << setw(15) << (eos.entropy(la, 0.2700001)  - eos.entropy(la, 0.2699999))/eos.entropy(la, 0.2700001);
		fout << endl;
	  }
  }
  fout.close();
}
