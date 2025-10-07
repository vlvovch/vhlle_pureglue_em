# vHLLE EM (Hydro + EM Probes)

Hydrodynamic simulation of heavy-ion collisions using the original vHLLE code (https://github.com/yukarpenko/vhlle), adapted to incorporate a time‑dependent quark‑undersaturated equation of state (EoS) and the emission of electromagnetic probes (photons and dileptons). The implementation is based on V. Vovchenko et al., [Phys. Rev. C 94, 024906 (2016)](https://doi.org/10.1103/PhysRevC.94.024906) [[arXiv:1604.06346](https://arxiv.org/abs/1604.06346)].

## Key Differences from vHLLE
- vHLLE implemented as a reusable library for linking.
- Time‑dependent EoS that depends on the fluid cell proper time.
- Evolution equation for the fluid cell proper time.
- Equation of state with time‑dependent quark‑undersaturated phase, as a linear combination of lattice QCD equations of state for pure SU(3) gauge theory and QCD with quarks (2+1 flavor). See the tables in `eos/Lattice_BW_YM.dat` and `eos/Lattice_BW_QGP.dat`.
- Dependency on ROOT largely eliminated.
- Electromagnetic probes computed throughout the evolution (optional CUDA acceleration for photons).

Note: the codebase is a bit old and rough and may be incomplete in places. It is provided as is, for reference.


## Usage (hlle_em)

Requirements: CMake (>= 3.18), a C++17 compiler (older may work, but not tested), optional CUDA Toolkit for GPU photon rates.

Build example:
```bash
mkdir build
cd build
cmake -S .. -B .
cmake --build . -j
```
By default, the build config is `RelWithDebInfo`. You can change it via `-DCMAKE_BUILD_TYPE=Debug|Release|RelWithDebInfo`.

To run the code, you need to provide a run folder with the necessary input files (e.g., `run_folder/photonsparams.dat`).
As an example, you can use the `run_folder` provided in the repository as a working directory and run `hlle_em` there.
For instance, to calculate observables for 0-20% centrality (2.76 TeV) and quark equilibration time of 1 fm/c, run:
```bash
cd ..  # Go back to the root of the repository
cd run_folder
../build/hlle_em 0-20.tau0-0-1.taus-1.interpolation
```
This computes the spectra of thermal direct photons and dileptons from the hydrodynamic evolution. To compare with experimental data or published results, you typically need to add prompt photons separately (not included here).
The code should run and out the results in `run_folder/hydro_output/0-20.tau0-0-1.taus-1.interpolation` folder.
It will compute photons from undersaturated QGP only (at T > 155 MeV) using AMY emission model, and from QGP (T > 155 MeV) plus hadronic phase (125 < T < 155 MeV).

Files in `run_folder`:
- `0-20.tau0-0-1.taus-1.interpolation` - input parameter file (also for other centralities and quark equilibration time)
- `photonsparams.dat` - photons spectrum parameters (QGP/AMY cut-off temperature, hadronic cut-off temperature, and list of pT bins)
- `dileptonsparams.dat` - dileptons parameters (cut-off temperature and list of invariant mass bins)
- `ic/MCGlauberIC.LHC/0-20.tb` - initial conditions (also for other centralities)

## Attribution and Citation

When using this code, please cite both (INSPIRE-HEP journal refs):
- I. Karpenko, P. Huovinen, M. Bleicher, "A 3+1 dimensional viscous hydrodynamic code for relativistic heavy ion collisions," [Comput. Phys. Commun. 185 (2014) 3016–3027](https://doi.org/10.1016/j.cpc.2014.07.010) [[arXiv:1312.4160](https://arxiv.org/abs/1312.4160)].
- V. Vovchenko, Iu. A. Karpenko, M. I. Gorenstein, L. M. Satarov, I. N. Mishustin, B. Kämpfer, H. Stoecker, "Electromagnetic probes of a pure-glue initial state in nucleus-nucleus collisions at the CERN LHC," [Phys. Rev. C 94 (2016) 024906](https://doi.org/10.1103/PhysRevC.94.024906) [[arXiv:1604.06346](https://arxiv.org/abs/1604.06346)].

If you modify the code and plan to publish results, the original vHLLE notice requests contacting the authors.


## Licensing

- New and modified code in this repository is licensed under the MIT License. See `LICENSE`.
- Third‑party components are included under their original terms. See `THIRD_PARTY_NOTICES` and the headers in the corresponding source files.

## Contact

For questions, issues, or guidance on using the code, please contact the author (V. Vovchenko). You can reach me at vvovchenko@uh.edu. Feel free to open an issue for tracking.
