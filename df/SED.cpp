#include <cmath>
#define ARMA_DONT_USE_WRAPPER
#include <armadillo>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>

#include "SED.h"

using namespace arma;
using namespace std;

void SED::extract_spec(const std::vector<std::vector<double>>& full_table,
                       const std::vector<double>& time_grid,
                       int a_indx, float ages4,
                       std::vector<long double>& spe) {
  // Interpolating between two spectra of contiguous ages not needed in cb16
  spe.clear();
  spe.reserve(13391);
  if (ages4 <= 0.0f) {
    const auto& f1 = full_table[1];
    for (int i = 0; i < 13391; i++) {
      spe.push_back(f1[i] * ergsa);
    }
  } else if (ages4 >= 20.0f) {
    const auto& f221 = full_table[221];
    for (int i = 0; i < 13391; i++) {
      spe.push_back(f221[i] * ergsa);
    }
  } else if (ages4 > 0.0f && ages4 < 20.0f) {
    const auto& fa = full_table[a_indx + 1];
    for (int i = 0; i < 13391; i++) {
      spe.push_back(fa[i] * ergsa);
    }
  }
}

double SED::getY(const std::vector<double>& x, const std::vector<double>& y, double xi) {
  int nn = (int)x.size();
  if (x[0] < x[nn - 1]) {
    if (xi > x[nn - 1]) return y[nn - 1];
    if (xi < x[0]) return y[0];
  } else {
    if (xi < x[nn - 1]) return y[nn - 1];
    if (xi > x[0]) return y[0];
  }
  int i = locate(x, xi);
  i = std::min(std::max(i, 0), nn - 2);
  double f = (xi - x[i]) / (x[i + 1] - x[i]);
  if (i > 1 && i < nn - 2) {
    double f2 = f * f;
    double a0 = y[i + 2] - y[i + 1] - y[i - 1] + y[i];
    double a1 = y[i - 1] - y[i] - a0;
    double a2 = y[i + 1] - y[i - 1];
    double a3 = y[i];
    return a0 * f * f2 + a1 * f2 + a2 * f + a3;
  } else {
    return f * y[i + 1] + (1.0 - f) * y[i];
  }
}

template <class T>
int SED::locate(const std::vector<T>& v, const T x) {
  size_t n = v.size();
  int jl = -1;
  int ju = (int)n;
  bool as = (v[n - 1] >= v[0]);
  while (ju - jl > 1) {
    int jm = (ju + jl) / 2;
    if ((x >= v[jm]) == as)
      jl = jm;
    else
      ju = jm;
  }
  if (x == v[0])
    return 0;
  else if (x == v[n - 1])
    return (int)n - 2;
  else
    return jl;
}

template int SED::locate<double>(const std::vector<double>&, const double);
template int SED::locate<float>(const std::vector<float>&, const float);
template int SED::locate<long double>(const std::vector<long double>&, const long double);

void SED::z_evol(float zr,
                 std::vector<long double>& wave,
                 std::vector<long double>& sed,
                 const std::vector<double>& zl,
                 const std::vector<double>& dlum) {
  double dls = 0.0;
  if (zr > 0.0f) {
    double dlsmpc = SED::getY(zl, dlum, zr);
    dls = dlsmpc * mpctocm / 0.6774;
  } else if (zr == 0.0f) {
    dls = 3.086e+19; // 10 pc to cm
  }

  // Redshifting and computing F(lambda/AA) from input L(lambda/AA) in-place
  const auto z_factor = (1 + zr);
  const auto scale_denom = (1 + zr) * pow(dls, 2) * 4 * M_PI;

  for (size_t l = 0; l < wave.size(); ++l) {
    wave[l] = wave[l] * z_factor;
    sed[l] = sed[l] / scale_denom;
  }
}

FilterPrecomp SED::precomputeFilter(const std::string& name,
                                    const std::vector<long double>& fwaves,
                                    const std::vector<long double>& fresp) {
  FilterPrecomp fp;
  fp.name = name;

  // find lambda_min and lambda_max of the filter with a threshold (1e-4)
  for (size_t i = 0; i < fresp.size(); ++i) {
    if (fresp[i] >= 1.e-4) {
      fp.lminf = (double)fwaves[i];
      break;
    }
  }

  for (size_t i = fresp.size(); i-- != 0;) {
    if (fresp[i] >= 1.e-4) {
      fp.lmaxf = (double)fwaves[i];
      break;
    }
  }

  fp.size_wave = (int)(2 * (fp.lmaxf - fp.lminf));
  fp.h_step = (fp.lmaxf - fp.lminf) / (double)fp.size_wave;

  std::vector<double> wave_i(fp.size_wave);
  double val = fp.lminf;
  for (size_t k = 0; k < (size_t)fp.size_wave; ++k) {
    wave_i[k] = val;
    val += fp.h_step;
  }
  fp.cwaves = arma::conv_to<arma::vec>::from(wave_i);

  arma::vec cfresp = arma::conv_to<arma::vec>::from(fresp);
  arma::vec cfwaves = arma::conv_to<arma::vec>::from(fwaves);

  arma::interp1(cfwaves, cfresp, fp.cwaves, fp.filter_interp, "*linear", cfresp[0]);
  fp.filter_cwaves = fp.filter_interp % fp.cwaves;

  arma::vec i2 = fp.filter_interp % (1.0 / fp.cwaves);
  arma::mat I2_mat = arma::trapz(fp.cwaves, i2);
  fp.I2 = arma::as_scalar(I2_mat);

  return fp;
}

double SED::compute_mab_fast(const std::vector<long double>& waves,
                             const std::vector<long double>& sed,
                             const FilterPrecomp& filter) {
  size_t start_idx = 0;
  while (start_idx < waves.size() && waves[start_idx] < filter.lminf) {
    ++start_idx;
  }
  size_t end_idx = start_idx;
  while (end_idx < waves.size() && waves[end_idx] <= filter.lmaxf) {
    ++end_idx;
  }

  size_t n_pts = (end_idx > start_idx) ? (end_idx - start_idx) : 0;
  if (n_pts == 0) {
    return 99.0;
  }

  arma::vec nwave(n_pts);
  arma::vec nsed(n_pts);
  for (size_t k = 0; k < n_pts; ++k) {
    nwave[k] = (double)waves[start_idx + k];
    nsed[k] = (double)sed[start_idx + k];
  }

  arma::vec sed_interp;
  arma::interp1(nwave, nsed, filter.cwaves, sed_interp, "*linear", 0.0);

  // Exact formula matching original:
  // filterSpec = filter_interp % csed;
  // i1 = filterSpec % cwaves;
  arma::vec filterSpec = filter.filter_interp % sed_interp;
  arma::vec i1 = filterSpec % filter.cwaves;
  arma::mat I1 = arma::trapz(filter.cwaves, i1);

  double flambda = arma::as_scalar(I1 / filter.I2);
  double fnu = flambda / speedcunitas;
  double mAB = -2.5 * log10(fnu) - 48.6;
  return mAB;
}

double SED::compute_mab(float zr,
                        std::vector<long double>& waves,
                        std::vector<long double>& sed,
                        std::vector<long double>& fwaves,
                        std::vector<long double>& fresp) {
  double lminf = 0.0, lmaxf = 0.0;
  for (size_t i = 0; i < fresp.size(); ++i) {
    if (fresp[i] >= 1.e-4) {
      lminf = fwaves[i];
      break;
    }
  }
  for (size_t i = fresp.size(); i-- != 0;) {
    if (fresp[i] >= 1.e-4) {
      lmaxf = fwaves[i];
      break;
    }
  }

  arma::vec cfresp = arma::conv_to<vec>::from(fresp);
  arma::vec cfwaves = arma::conv_to<vec>::from(fwaves);

  std::vector<long double> newwaves, newsed;
  for (size_t i = 0; i < sed.size(); ++i) {
    if ((waves[i] >= lminf) && (waves[i] <= lmaxf)) {
      newwaves.push_back(waves[i]);
      newsed.push_back(sed[i]);
    }
  }

  int size_wave = (int)(2 * (lmaxf - lminf));
  double h_step = (lmaxf - lminf) / (size_wave);

  std::vector<double> wave_i(size_wave);
  double val = lminf;
  for (size_t k = 0; k < (size_t)size_wave; ++k) {
    wave_i[k] = val;
    val += h_step;
  }

  arma::vec wave_interp = arma::conv_to<vec>::from(wave_i);
  arma::vec nsed = arma::conv_to<vec>::from(newsed);
  arma::vec nwave = arma::conv_to<vec>::from(newwaves);
  arma::vec sed_interp;

  arma::interp1(nwave, nsed, wave_interp, sed_interp, "*linear", 0);

  arma::vec csed = sed_interp;
  arma::vec cwaves = wave_interp;
  arma::vec filter_interp;

  arma::interp1(cfwaves, cfresp, cwaves, filter_interp, "*linear", cfresp[0]);

  arma::vec filterSpec = filter_interp % csed;
  arma::vec i1 = filterSpec % cwaves;
  arma::vec i2 = filter_interp % (1.0 / cwaves);

  arma::mat I1 = arma::trapz(cwaves, i1);
  arma::mat I2 = arma::trapz(cwaves, i2);

  double flambda = arma::as_scalar(I1 / I2);
  double fnu = flambda / speedcunitas;
  double mAB = -2.5 * log10(fnu) - 48.6;

  return mAB;
}

void SED::dust_attenuation(std::vector<float>& ex_l,
                           std::vector<long double>& dustAtt,
                           std::vector<long double>& wavesc,
                           std::vector<long double>& spe) {
  arma::vec ex_l_interp = arma::conv_to<vec>::from(wavesc);
  arma::vec a_spe = arma::conv_to<vec>::from(spe);
  arma::vec a_ex_l = arma::conv_to<vec>::from(ex_l);
  arma::vec a_dustAtt = arma::conv_to<vec>::from(dustAtt);
  arma::vec dustAtt_interp;

  arma::interp1(a_ex_l, a_dustAtt, ex_l_interp, dustAtt_interp, "*linear", 1);

  arma::vec out_spe = dustAtt_interp % a_spe;
  spe = arma::conv_to<std::vector<long double>>::from(out_spe);
  dustAtt = arma::conv_to<std::vector<long double>>::from(dustAtt_interp);
  ex_l = arma::conv_to<std::vector<float>>::from(ex_l_interp);
}
