#ifndef SED_H
#define SED_H

#include <cmath>
#define ARMA_DONT_USE_WRAPPER
#include <armadillo>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>

struct FilterPrecomp {
  std::string name;
  double lminf = 0.0;
  double lmaxf = 0.0;
  int size_wave = 0;
  double h_step = 0.0;
  arma::vec cwaves;
  arma::vec filter_interp;
  arma::vec filter_cwaves;
  double I2 = 0.0;
};

class SED
{
public:
  SED() = default;

  void extract_spec(const std::vector<std::vector<double>>& full_table,
                    const std::vector<double>& time_grid,
                    int a_indx, float ages4,
                    std::vector<long double>& spe);
  
  double getY(const std::vector<double>& x, const std::vector<double>& y, double xi);

  double compute_mab(float zr,
                     std::vector<long double>& waves,
                     std::vector<long double>& sed,
                     std::vector<long double>& fwaves,
                     std::vector<long double>& fresp);

  static FilterPrecomp precomputeFilter(const std::string& name,
                                       const std::vector<long double>& fwaves,
                                       const std::vector<long double>& fresp);

  double compute_mab_fast(const std::vector<long double>& waves,
                          const std::vector<long double>& sed,
                          const FilterPrecomp& filter);

  void z_evol(float zr,
              std::vector<long double>& wave,
              std::vector<long double>& sed,
              const std::vector<double>& zl,
              const std::vector<double>& dlum);

  void dust_attenuation(std::vector<float>& ex_l,
                        std::vector<long double>& dustAtt,
                        std::vector<long double>& wavesc,
                        std::vector<long double>& spe);

  template <class T> int locate(const std::vector<T>& v, const T x);

  ~SED() = default;

private:
  const double ergsa = 3.9e+33;
  const double speedcunitas = 2.9979e+18; // AA/s
  const double mpctocm = 3.086e+24;
};

#endif // SED_H
