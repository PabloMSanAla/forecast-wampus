#include <cmath> 
#include <chrono> 
#include <iostream>
#include <string>
#include <filesystem>
#include <sstream>
#include <fstream>
#include <vector>
#include <iterator>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <cstdio>     
#include <cstdlib>     
#include <ctime>
#include <unordered_map>
#include <cassert>
#include <sys/resource.h>
#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>
#endif
#include <H5Cpp.h>
#include <Eigen/Dense>
#define ARMA_DONT_USE_WRAPPER
#include <armadillo>

#include "readTNGParticle.h"
#include "SED.h"
#include "functions.h"

/*****************************************************************************/
/*                                                                           */
/*             FORECAST - dust-corrected fluxes calculations module          */
/*                                                                           */
/*  original dark matter-only code by cgiocoli@gmail.com                     */
/*  updated to its final form by flaminia.fortuni@inaf.it                    */
/*  optimized and RAM-aware streaming engine (2026)                          */
/*                                                                           */
/*****************************************************************************/

using namespace std;
using namespace std::chrono;
using namespace arma;

const double speedcunit = 2.99792458e+3;
const double mpM = 8.4089382e-58; // proton mass in Msun
const double mpg = 1.6726219e-24; // proton mass in g
const double mpc2tocm2 = 9.523e+42; // Mpc^2 to cm^2
const double h0 = 0.6774;

// 4th grade fit coefficients (Nelson+19 / Fortuni+23)
const float a1 = 2.31105169e-02f;
const float a2 = -8.91893982e-01f;
const float a3 = 1.14885315e+01f;
const float a4 = -6.27073890e+01f;
const float a5 = 1.18956375e+02f;

// Darwin-aware RSS monitoring: returns peak RSS in MB
inline double getPeakRSS_MB() {
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
#if defined(__APPLE__) && defined(__MACH__)
  // macOS returns ru_maxrss in bytes
  return (double)usage.ru_maxrss / (1024.0 * 1024.0);
#else
  // Linux returns ru_maxrss in kilobytes
  return (double)usage.ru_maxrss / 1024.0;
#endif
}

inline void checkMemoryCeiling(double ceiling_gb) {
  double current_peak_mb = getPeakRSS_MB();
  double ceiling_mb = ceiling_gb * 1024.0;
  if (current_peak_mb > ceiling_mb) {
    cerr << "Error: Memory ceiling exceeded! Peak RSS: " << current_peak_mb
         << " MB > Limit: " << ceiling_mb << " MB (" << ceiling_gb << " GB)" << endl;
    exit(3);
  }
}

int main(int argc, char** argv) {
  auto start = high_resolution_clock::now();

  cout << "----------------------------------------------" << endl;
  cout << "-                                            -" << endl;
  cout << "::::DUST POST-PROCESSING for the LIGHTCONE::::" << endl;
  cout << "-                                            -" << endl;
  cout << "----------------------------------------------" << endl;

  // 1. CLI Parsing & Validation
  int sourceID = -1;
  int iplrestart = -1;
  double memory_ceiling_gb = 4.0; // Default 4GB
  string custom_ini = "";
  vector<string> positional_args;

  for (int i = 1; i < argc; i++) {
    string arg = argv[i];
    if (arg == "-ini") {
      if (i + 1 < argc) {
        custom_ini = argv[++i];
      } else {
        cerr << "Error: -ini option requires a file path." << endl;
        exit(1);
      }
    } else if (arg == "-h" || arg == "--help") {
      cout << "Usage: " << argv[0] << " <snapshot> <plane_number> [memory_ceiling_gb] [-ini <dc.ini>]" << endl;
      exit(0);
    } else {
      positional_args.push_back(arg);
    }
  }

  if (positional_args.size() < 2 || positional_args.size() > 3) {
    cerr << "Usage: " << argv[0] << " <snapshot> <plane_number> [memory_ceiling_gb] [-ini <dc.ini>]" << endl;
    exit(1);
  }

  try {
    sourceID = std::stoi(positional_args[0]);
    iplrestart = std::stoi(positional_args[1]);
  } catch (...) {
    cerr << "Error: Both snapshot and plane number must be integers." << endl;
    exit(1);
  }

  if (sourceID < 0 || iplrestart < 0) {
    cerr << "Error: Snapshot and plane number must be non-negative integers." << endl;
    exit(1);
  }

  if (positional_args.size() == 3) {
    try {
      memory_ceiling_gb = std::stod(positional_args[2]);
      if (memory_ceiling_gb <= 0.0) {
        cerr << "Error: Memory ceiling must be a positive number." << endl;
        exit(1);
      }
    } catch (...) {
      cerr << "Error: Memory ceiling must be a valid number." << endl;
      exit(1);
    }
  }

  cout << "Initialize DC module for plane number: " << iplrestart 
       << ", snapshot: " << sourceID 
       << ", memory ceiling: " << memory_ceiling_gb << " GB";
  if (!custom_ini.empty()) {
    cout << ", ini file: " << custom_ini;
  }
  cout << endl;

  // Initial ceiling check
  checkMemoryCeiling(memory_ceiling_gb);

  // 2. Read Configuration from dc.ini
  double boxl;
  float maglim;
  string filfilters, filsnaplist, filtimelist, idc;
  string pathsnap, bc03dir, dfpath, rdir; 
  int read;
  string model, imf;
  string filextc;
  string planes_file_ini;

  readParameters(&boxl, &maglim,
                 &filfilters, &filsnaplist, &filtimelist, &idc,
                 &pathsnap, &bc03dir, &dfpath, &rdir,
                 &read,
                 &model, &imf,
                 &filextc,
                 &planes_file_ini,
                 custom_ini);

  // 3. Resolve planes_list.txt
  string fplane = "";
  const char* env_planes = std::getenv("FORECAST_PLANES_LIST");
  if (env_planes != nullptr && env_planes[0] != '\0') {
    fplane = env_planes;
  } else if (!planes_file_ini.empty()) {
    fplane = planes_file_ini;
  } else {
    const char* fallback_paths[] = {
      "planes_list.txt",
      "dc/planes_list.txt",
      "../lc/planes_list.txt",
      "lc/planes_list.txt",
      "/Volumes/PMSAdrive/pmsa/storage/TNG/optimization/planes_list.txt"
    };
    for (const char* p : fallback_paths) {
      ifstream tf(p);
      if (tf.is_open()) {
        fplane = p;
        tf.close();
        break;
      }
    }
  }

  ifstream oplane;
  if (!fplane.empty()) {
    oplane.open(fplane.c_str());
  }
  if (!oplane.is_open()) {
    cerr << "Error: planes_list.txt could not be found. Please set PLANES_LIST_FILE in dc.ini or FORECAST_PLANES_LIST env var." << endl;
    exit(2);
  }

  int n_pl = 0;
  float blD = 0.0f, blD2 = 0.0f, zsim = 0.0f;
  string snappl = "";
  bool plane_matched = false;

  int npl, reppl, blsnappl;
  float zpl, blDpl, blD2pl, zpltrue;

  while (oplane >> npl >> zpl >> blDpl >> blD2pl >> reppl >> blsnappl >> zpltrue) {
    if (blsnappl == sourceID && (reppl == iplrestart || npl - 1 == iplrestart)) {
      n_pl = npl;
      blD = blDpl;
      blD2 = blD2pl;
      snappl = conv(blsnappl, fINT);
      zsim = zpltrue;
      plane_matched = true;
      break;
    }
  }
  oplane.close();

  if (!plane_matched) {
    cerr << "Error: Snapshot " << sourceID << " and plane " << iplrestart
         << " do not match any entry in " << fplane << "." << endl;
    exit(1);
  }

  // 4. Load Timelist
  ifstream timelist(filtimelist.c_str());
  if (!timelist.is_open()) {
    cerr << "Error: time list file " << filtimelist << " could not be found. Check FILE_TIMELIST in dc.ini." << endl;
    exit(2);
  }
  vector<int> snap_tlist;
  vector<double> z_tlist;
  vector<double> age_tlist;
  int buta;
  double butb, butc;
  while (timelist >> buta >> butb >> butc) {
    snap_tlist.push_back(buta);
    z_tlist.push_back(butb);
    age_tlist.push_back(butc);
  }
  timelist.close();

  // 5. Load Age Table (age_bc03.txt or age_cb16.txt)
  string filagelist = "../files/age_" + model + ".txt";
  ifstream agelist(filagelist.c_str());
  if (!agelist.is_open()) {
    filagelist = "files/age_" + model + ".txt";
    agelist.open(filagelist.c_str());
  }
  if (!agelist.is_open()) {
    filagelist = "age_" + model + ".txt";
    agelist.open(filagelist.c_str());
  }
  if (!agelist.is_open()) {
    filagelist = rdir + "age_" + model + ".txt";
    agelist.open(filagelist.c_str());
  }
  if (!agelist.is_open()) {
    filagelist = "age_" + model + ".txt";
    agelist.open(filagelist.c_str());
  }
  if (!agelist.is_open()) {
    cerr << "Error: age list file for model " << model << " could not be found." << endl;
    exit(2);
  }
  vector<double> nage;
  vector<long double> age_bc03;
  double age_a, age_b;
  while (agelist >> age_a >> age_b) {
    nage.push_back(age_a);
    age_bc03.push_back(age_b);
  }
  agelist.close();

  // 6. Load Extinction Curve from EXTINCTION_FILE
  ifstream extcurve(filextc.c_str());
  if (!extcurve.is_open() && filextc.rfind("../", 0) == 0) {
    string alt_filextc = filextc.substr(3);
    extcurve.open(alt_filextc.c_str());
    if (extcurve.is_open()) filextc = alt_filextc;
  }
  if (!extcurve.is_open() && filextc.rfind("dc/", 0) == 0) {
    string alt_filextc = "../" + filextc;
    extcurve.open(alt_filextc.c_str());
    if (extcurve.is_open()) filextc = alt_filextc;
  }
  if (!extcurve.is_open()) {
    cerr << "Error: Extinction curve file " << filextc << " could not be found. Check EXTINCTION_FILE in dc.ini." << endl;
    exit(2);
  }
  vector<long double> ex_l;
  vector<long double> ex_curve;
  double Rv = 3.1;
  string str1;
  std::getline(extcurve, str1); // skip 1 header line
  float ex_a, ex_b, ex_c, ex_d, ex_e;
  while (extcurve >> ex_a >> ex_b >> ex_c >> ex_d >> ex_e) {
    ex_l.push_back(ex_a * 1.e4); // [um] to [AA]
    ex_curve.push_back(ex_b + ex_c / Rv);
  }
  extcurve.close();

  // 7. Load Cosmology Table
  ifstream infiledc(idc.c_str());
  if (!infiledc.is_open()) {
    string idc_fallback = "files/LCDM-comovingdistTNG.dat";
    infiledc.open(idc_fallback.c_str());
  }
  if (!infiledc.is_open()) {
    string idc_fallback = "../files/LCDM-comovingdistTNG.dat";
    infiledc.open(idc_fallback.c_str());
  }
  if (!infiledc.is_open()) {
    string idc_fallback = "/Users/pmsanch1/code/FORECAST/files/LCDM-comovingdistTNG.dat";
    infiledc.open(idc_fallback.c_str());
  }
  if (!infiledc.is_open()) {
    cerr << "Error: comoving distance file " << idc << " could not be found." << endl;
    exit(2);
  }
  vector<double> zl, dl, dlum;
  double zi, dci, dli;
  while (infiledc >> zi >> dci >> dli) {
    zl.push_back(zi);
    dl.push_back(dci * speedcunit);   // comoving distance [cMpc/h]
    dlum.push_back(dli * speedcunit); // luminosity distance [cMpc/h]
  }
  infiledc.close();

  // 8. Load Filters
  ifstream listfilter(filfilters.c_str());
  if (!listfilter.is_open()) {
    string flt_fallback = "files/filters.dat";
    listfilter.open(flt_fallback.c_str());
  }
  if (!listfilter.is_open()) {
    string flt_fallback = "../files/filters.dat";
    listfilter.open(flt_fallback.c_str());
  }
  if (!listfilter.is_open()) {
    string flt_fallback = "/Users/pmsanch1/code/FORECAST/files/filters.dat";
    listfilter.open(flt_fallback.c_str());
  }
  if (!listfilter.is_open()) {
    cerr << "Error: filter list file " << filfilters << " could not be found." << endl;
    exit(2);
  }
  vector<string> nfilter;
  string ftr_name;
  while (listfilter >> ftr_name) {
    nfilter.push_back(ftr_name);
  }
  listfilter.close();

  int Nfilters = nfilter.size();
  std::vector<vector<long double>> fresp(Nfilters, vector<long double>(0));
  std::vector<vector<long double>> fwaves(Nfilters, vector<long double>(0));

  string filter_dir = "../files/filters/";
  {
    ifstream test_f((filter_dir + nfilter[0] + ".dat").c_str());
    if (!test_f.is_open()) {
      filter_dir = "files/filters/";
      ifstream test_f2((filter_dir + nfilter[0] + ".dat").c_str());
      if (!test_f2.is_open()) {
        filter_dir = std::filesystem::path(filfilters).parent_path().string() + "/filters/";
      }
    }
  }

  for (int N = 0; N < Nfilters; N++) {
    string filterin = filter_dir + nfilter[N] + ".dat";
    ifstream filterlist(filterin.c_str());
    if (!filterlist.is_open()) {
      cerr << "Error: Filter file " << filterin << " is missing." << endl;
      exit(2);
    }
    long double sa, sb;
    while (filterlist >> sa >> sb) {
      fwaves[N].push_back(sa);
      fresp[N].push_back(sb);
    }
    filterlist.close();
  }

  // 9. Load SSP Tables
  std::vector<long double> waves; // [AA]
  vector<vector<double>> time_grid;
  vector<vector<vector<double>>> full_table;
  if (model == "bc03") {
    time_grid.resize(6);
    full_table.resize(6);
    for (int i = 0; i < 6; ++i) {
      readSSPTables(bc03dir, model, imf, i, waves, time_grid[i], full_table[i]);
    }
  } else if (model == "cb16") {
    time_grid.resize(14);
    full_table.resize(14);
    for (int i = 0; i < 14; ++i) {
      readSSPTables(bc03dir, model, imf, i, waves, time_grid[i], full_table[i]);
    }
  }

  // Constants & Parameters
  double bs, om0, omL0, dlsim;
  vector<long double> met_bc03 { 0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05 };
  vector<long double> met_cb16 { 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.004, 0.006, 0.008, 0.010, 0.014, 0.017, 0.020, 0.030, 0.040 };
  SED sedy; 
  float convrho = (1.e10 / h0) / (pow(1. / h0, 3.));
  float convm = 1.e10 / h0;
  float convZsun = 1. / 0.0127;
  double invmpM = 1. / mpM;   // [Msun-1]
  double kB = 1.38e-16;     // [erg/K]
  double gamma = 5. / 3.;     // adiabatic index
  double xH = 0.76;         // H fraction
  double NHI_0 = 2.e64;     // [kpc-2]
  double invNHI_0 = 1. / NHI_0;

  cout << endl;
  cout << "... Reading dust-free particle-based catalog produced in the previous module (flux.df.snap_plane.txt) ..." << endl;
  cout << " .. content: #(ids,x,y,redshift,mass,intial mass,metallicity, CM_sh, age, N_sim) for stellar particles;" << endl;
  cout << " ..          #(dust-free flux in Nfilters) for stellar particles." << endl;
  cout << endl;

  // 10. Pass 1: Stream DF Catalog & Aggregate Subhalo Properties Online
  if (!dfpath.empty() && dfpath.back() != '/') {
    dfpath += "/";
  }
  string filoutcat = dfpath + "flux.df." + snappl + "_" + conv(iplrestart, fINT) + ".txt";
  ifstream ocf(filoutcat.c_str());
  if (!ocf.is_open()) {
    cerr << "Error: OUT_CAT file " << filoutcat << " does not exist for this snapshot." << endl;
    exit(2);
  }

  // Pre-allocate 64KB I/O buffer for fast streaming reading
  char in_buf_df[65536];
  ocf.rdbuf()->pubsetbuf(in_buf_df, sizeof(in_buf_df));

  std::vector<int> idsh;
  std::unordered_map<int, size_t> sh_to_idx;
  std::vector<int> Npsh;
  std::vector<float> zrsh;
  std::vector<float> CMzsh;
  std::vector<std::vector<float>> fluxsh;
  std::vector<float> ms4sh;
  std::vector<std::vector<long double>> galSpe;
  std::vector<int> NpshTNG_sh;
  std::vector<int> NpshTNG_first;

  size_t totPartMapxy4 = 0;
  int shid;
  float xi, yi, zri, mi, imi, ai, cmzsh;
  long double meti, NPTNG;

  while (ocf >> shid >> xi >> yi >> zri >> mi >> imi >> meti >> cmzsh >> ai >> NPTNG) { 
    if (NpshTNG_first.size() < 1000000) {
      NpshTNG_first.push_back(static_cast<int>(NPTNG));
    } 
    std::vector<double> temp_flux(Nfilters, 0.0);
    for (int cf = 0; cf < Nfilters; ++cf) {
      if (!(ocf >> temp_flux[cf])) break;
    }
    ocf.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    auto it = sh_to_idx.find(shid);
    size_t k;
    if (it == sh_to_idx.end()) {
      k = idsh.size();
      idsh.push_back(shid);
      sh_to_idx[shid] = k;
      Npsh.push_back(0);
      zrsh.push_back(0.0f);
      CMzsh.push_back(cmzsh);
      ms4sh.push_back(0.0f);
      fluxsh.push_back(std::vector<float>(Nfilters, 0.0f));
      galSpe.push_back(std::vector<long double>(waves.size(), 0.0L));
      NpshTNG_sh.push_back(static_cast<int>(NPTNG));
    } else {
      k = it->second;
    }

    Npsh[k]++;
    zrsh[k] += zri;
    CMzsh[k] = cmzsh;
    ms4sh[k] += mi;
    for (int cf = 0; cf < Nfilters; ++cf) {
      fluxsh[k][cf] += temp_flux[cf];
    }

    // Accumulate unattenuated model B spectrum for subhalo online
    int age_inx = index_closest(age_bc03.begin(), age_bc03.end(), ai);
    int met_inx;
    std::vector<long double> spe;
    if (model == "bc03") {
      met_inx = index_closest(met_bc03.begin(), met_bc03.end(), meti);
      SEDbc03_interp_2spec(full_table[met_inx], time_grid[met_inx], age_inx, ai, spe);
    } else if (model == "cb16") {
      met_inx = index_closest(met_cb16.begin(), met_cb16.end(), meti);
      SEDcb16_extract_spec(full_table[met_inx], time_grid[met_inx], age_inx, ai, spe);
    }

    for (size_t i = 0; i < waves.size(); ++i) {
      long double modB_val = (ai <= 0.01f) ? expl(-1.0 * pow((waves[i] / 5500.0), -0.7)) : expl(-0.3 * pow((waves[i] / 5500.0), -0.7));
      galSpe[k][i] += (spe[i] * modB_val) * imi;
    }

    totPartMapxy4++;
  }
  ocf.close();

  size_t num_sh = idsh.size();
  for (size_t k = 0; k < num_sh; ++k) {
    if (Npsh[k] > 0) {
      zrsh[k] /= Npsh[k];
    }
  }

  cout << totPartMapxy4 << "   type (4)   - STAR particles in dust-free catalogue " << endl;
  cout << endl; 
  cout << "... Some parameters ..." << endl;
  cout << " (D_l, D_l2) = (" << blD << ", " << blD2 << ") cMpc; z between (" << (getY(dl, zl, blD)) << ", " << (getY(dl, zl, blD2)) << "); snapshot " << sourceID << ", plane " << n_pl << endl;
  cout << " filters: " << endl;
  for (size_t i = 0; i < nfilter.size(); i++) {
    cout << nfilter[i] << "  -  ";
  }
  cout << endl << "....................... " << endl << endl;

  cout << "... Creating SH cat from input dust-free star particles cat ..." << endl;
  cout << endl;
  cout << "... Starting to read TNG cat ..." << endl;
  cout << endl;

  std::vector<double> NHIm, Zm;

  if (read == 1) {
    checkMemoryCeiling(memory_ceiling_gb);

    // TNG particle class
    readTNGParticle tngParticle;
    tngParticle.Initialize(pathsnap, sourceID);
    tngParticle.readHeader(0); // chunk file id:0   
    bs = tngParticle.getBoxSize(); // [ckpc/h]
    om0 = tngParticle.getOmegaZero();
    omL0 = tngParticle.getOmegaLambda();
    zsim = tngParticle.getRedshift();
    dlsim = getY(zl, dl, zsim); // [Mpc/h]
    std::vector<int> npart = tngParticle.getNumPartTotal();
    cout << endl;

    std::string workdir_sn = pathsnap + "/snapdir_0" + conv(sourceID, fINT) + "/";
    std::string workdir_g = pathsnap + "/groups_0" + conv(sourceID, fINT) + "/";
    std::string workdir_os = pathsnap + "/offsets_0" + conv(sourceID, fINT) + ".hdf5";
    int nf_sn = countHDF5Files(workdir_sn, "hdf5");
    int nf_g = countHDF5Files(workdir_g, "hdf5");

    cout << " .. Extracting gas from TNG files .." << endl;

    vector<int> gcLenTypeG(0), gcOffsetsTypeG(0);
    for (int i = 0; i < nf_g; i++) {
      tngParticle.readFof(i);
      std::vector<int> sLTG = tngParticle.getGasLenType();
      gcLenTypeG.insert(gcLenTypeG.end(), sLTG.begin(), sLTG.end());
    }  
    tngParticle.readOffset();
    gcOffsetsTypeG = tngParticle.getGasByType();

    std::vector<double> NHIsum(num_sh, 0.0);
    std::vector<double> ZHIsum(num_sh, 0.0);
    std::vector<double> MHIsum(num_sh, 0.0);
    size_t idshg_count = 0;

    vector<int> diff = vec_diff(gcOffsetsTypeG);
    vector<int> gcOffsetsMax = vec_sum(gcOffsetsTypeG, gcLenTypeG);
    size_t num_gas_sub = gcOffsetsTypeG.size();
    int cur_sub = 0;
    int global_id = 0;

    for (int i = 0; i < nf_sn; i++) {
      tngParticle.readGAS(i);
      const vector<double>& metal0 = tngParticle.getGASMetallicity();
      const vector<double>& Mass0 = tngParticle.getGASMasses(); // [1e10 Msun/h]
      const vector<double>& Rho_0 = tngParticle.getGASDensity(); // [(1e10 Msun/h)/(kpc/h)^3]
      const vector<double>& Z_0 = tngParticle.getGASZ(); 	
      const vector<double>& u_0 = tngParticle.getGASIntEnergy();
      const vector<double>& x_e0 = tngParticle.getGASeAbundance();
      
      int metal_length = (int)metal0.size();
      
      for (int k = 0; k < metal_length; k++) {
        int pid = global_id + k;
        while (cur_sub + 1 < (int)num_gas_sub && pid > diff[cur_sub + 1]) {
          cur_sub++;
        }
        int sh_id = -1;
        if (pid >= gcOffsetsTypeG[0] && cur_sub < (int)num_gas_sub && pid <= gcOffsetsMax[cur_sub]) {
          sh_id = cur_sub;
        }

        if (sh_id > -1) {
          auto it = sh_to_idx.find(sh_id);
          if (it != sh_to_idx.end()) {
            size_t j = it->second;
            float mag0 = 2.5f * (29.0f - std::log10(fluxsh[j][0])) - 48.6f;
            float cmz_limit = (mag0 <= maglim) ? CMzsh[j] : -9.0f;
            idshg_count++;

            float mg_val = static_cast<float>(Mass0[k] * convm); // Msun

            if (Z_0[k] >= cmz_limit) {
              double V0 = (Rho_0[k] != 0.0) ? ((Mass0[k] * convm) / (Rho_0[k] * convrho)) : 0.0;
              double L0 = pow(V0, 1.0 / 3.0); // [ckpc]
              double mu0 = (4.0 * mpg) / (1.0 + 3.0 * 0.76 + 4.0 * 0.76 * x_e0[k]); // [g] and fixed xH=0.76
              double T0 = (gamma - 1.0) * (u_0[k] / kB) * 1.e10 * mu0; // [K]
              double logT0 = log10(T0);
              double logMratio = a1 * pow(logT0, 4.0) + a2 * pow(logT0, 3.0) + a3 * pow(logT0, 2.0) + a4 * logT0 + a5;
              double Mratio = pow(10.0, logMratio); 
              double nHIgInt = Mratio * Rho_0[k] * convrho * invmpM; // [kpc-3]
              double weight = mg_val * Mratio;
              
              NHIsum[j] += weight * (nHIgInt * L0);
              ZHIsum[j] += weight * metal0[k];
              MHIsum[j] += weight;
            }
          }
        }
      }
      global_id += metal_length;
    }

    checkMemoryCeiling(memory_ceiling_gb);

    cout << idshg_count << "   type (0)   - GAS  cells     firstly selected in the snapshot." << endl;
    cout << endl;

    auto stop2 = high_resolution_clock::now(); 
    auto duration2 = duration_cast<microseconds>(stop2 - start);
    cout << "execution time in [h] " << 2.77778e-10 * duration2.count() << endl;

    cout << endl;
    cout << totPartMapxy4 << "   type (4)  - STAR particles in dust-free catalogue." << endl;
    cout << endl;    
    cout << idshg_count << "        type (0) - GAS cells with same SH ids." << endl; 
    cout << endl; 
    cout << "... Selecting TNG gas cells in front of MapSim star particles ..." << endl;

    NHIm.reserve(num_sh);
    Zm.reserve(num_sh);

    for (size_t l = 0; l < num_sh; ++l) {
      if (MHIsum[l] != 0.0) {
        NHIm.push_back((NHIsum[l] / MHIsum[l]) * invNHI_0);
        Zm.push_back((ZHIsum[l] / MHIsum[l]) * convZsun);
      } else {
        NHIm.push_back(0.0);
        Zm.push_back(0.0);
      }
    }

    cout << "::: gas done ::: " << endl;
    cout << endl;
  } else if (read == 0) {
    bs = boxl * 1000.0;
    om0 = 0.3089;
    omL0 = 0.6911;
    dlsim = getY(zl, dl, zsim);

    cout << " .. Extracting gas properties from gas catalog (coords.dc.snap_plane.txt) .." << endl;
    cout << " .. ! This is an old version of coords.xx files ! .." << endl;
    cout << " .. content: #(ids,x,y,redshift,mass,intial mass,metallicity, age) for stellar particles;" << endl;
    cout << " ..          #(Zgas_w, NHIgas_w) computed on galaxy-basis, for galaxies with id==is." << endl;
    cout << " ..          #(dust-free flux in Nfilters) for stellar particles." << endl;
    cout << " ..          #(Np_sh, N_sim): number of particles with ids in the lightcone, number of particle with ids in the original simulation." << endl;
    cout << endl;

    vector<int> idsh0p, NpshTNGp;
    vector<double> Zmp, NHImp;
    if (!rdir.empty() && rdir.back() != '/') {
      rdir += "/";
    }
    string filoutcatgas = rdir + "coords.dc." + snappl + "_" + conv(iplrestart, fINT) + ".txt";
    ifstream ocfgas(filoutcatgas.c_str());
    if (ocfgas.is_open()) {
      string line;
      while (getline(ocfgas, line)) {
        stringstream ss(line);
        int shid, npsh, nptng;
        double ZM, NHIM;
        float xi, yi, zri, mi, imi, ai;
        long double meti;
        ss >> shid >> xi >> yi >> zri >> mi >> imi >> meti >> ai >> ZM >> NHIM;
        for (int i = 0; i < Nfilters; i++) {
          double temp;
          ss >> temp;
        }
        ss >> npsh >> nptng;
        idsh0p.push_back(shid);
        Zmp.push_back(ZM);
        NHImp.push_back(NHIM);
        NpshTNGp.push_back(nptng);
      }
      ocfgas.close();
    } else {
      cerr << "Error: OUT_CAT file " << filoutcatgas << " does not exist for this snapshot." << endl;
      exit(2);
    }
    
    std::unordered_map<int, size_t> gas_prop_map;
    for (size_t j = 0; j < idsh0p.size(); ++j) {
      if (gas_prop_map.find(idsh0p[j]) == gas_prop_map.end()) {
        gas_prop_map[idsh0p[j]] = j;
      }
    }
    for (size_t k = 0; k < num_sh; ++k) {
      auto it = gas_prop_map.find(idsh[k]);
      if (it != gas_prop_map.end()) {
        size_t j = it->second;
        Zm.push_back(Zmp[j]);
        NHIm.push_back(NHImp[j]);
      } else {
        Zm.push_back(0.0);
        NHIm.push_back(0.0);
      }
    }

    cout << "::: gas read ::: " << endl;
    cout << endl;
  }

  // 11. Flux Assignment & Dust Extinction per Subhalo
  cout << "... Now assigning fluxes ..." << endl;
  cout << endl;
  std::vector<vector<double>> flux, df;  
  flux.reserve(num_sh);
  df.reserve(num_sh);

  for (size_t k = 0; k < num_sh; ++k) {
    float zr = zrsh[k];    
    std::vector<long double> modC;
    std::vector<long double> wavesc = waves;
    std::vector<float> ex_ls(15);
    
    // Compute modC Nelson+19 with mean values   
    if (NHIm[k] != 0.0) {
      sedy.modelC(zr, ex_curve, ex_l, ex_ls, modC, NHIm[k], Zm[k]);
    } else {
      modC.assign(ex_curve.size(), 1.0L);
    }
    
    // Redshift evolution
    sedy.z_evol(zr, wavesc, galSpe[k], zl, dlum);
   
    // Dust attenuation
    sedy.dust_attenuation(ex_ls, modC, wavesc, galSpe[k]);
    
    // Computing apparent magnitude in chosen filter; flux in [uJy]
    vector<double> f;
    vector<double> dfn(Nfilters, 0.0);
    f.reserve(Nfilters);

    for (int N = 0; N < Nfilters; ++N) {
      double m_ = sedy.compute_mab(zr, wavesc, galSpe[k], fwaves[N], fresp[N]);
      double f_ = pow(10.0, (29.0 - (m_ + 48.6) / 2.5));
      f.push_back(f_);

      if (fluxsh[k][N] > f_) {
        dfn[N] = f_ / fluxsh[k][N];
      } else {
        dfn[N] = 1.0;
      }
    }
    
    flux.push_back(f);
    df.push_back(dfn);
  }

  // Free galSpe immediately to minimize memory footprint
  std::vector<std::vector<long double>>().swap(galSpe);

  checkMemoryCeiling(memory_ceiling_gb);

  auto stop3 = high_resolution_clock::now(); 
  auto duration3 = duration_cast<microseconds>(stop3 - start);
  cout << "execution time in [h] " << 2.77778e-10 * duration3.count() << endl;
  		
  cout << endl;
  cout << "... Reddening the flux particle by particle, for the image ..." << endl;

  // 12. Pass 2: Stream DF Catalog Directly to Final Outputs
  if (!rdir.empty() && rdir.back() != '/') {
    rdir += "/";
  }

  ofstream myfile2g;
  char out_buf_coords[65536];
  if (read == 1) {
    cout << " ... Writing gas catalog (coords.dc.snap_plane.txt) ..." << endl;
    cout << " .. content: #(ids,x,y,redshift,mass,intial mass,metallicity, age) for stellar particles;" << endl;
    cout << " ..          #(Zgas_w, NHIgas_w) computed on galaxy-basis, for galaxies with id==is." << endl;
    cout << " ..          #(dust-free flux in Nfilters) for stellar particles." << endl;
    cout << endl;

    string gcoord_path = rdir + "coords.dc." + snappl + "_" + conv(iplrestart, fINT) + ".txt";
    myfile2g.rdbuf()->pubsetbuf(out_buf_coords, sizeof(out_buf_coords));
    myfile2g.open(gcoord_path.c_str());
    if (!myfile2g.is_open()) {
      cerr << "Error: Could not open " << gcoord_path << " for writing." << endl;
      exit(2);
    }
  }

  cout << "... Writing output file (flux.dc.snap_plane.txt) from this module ..." << endl;
  cout << " .. content: #(ids,x,y,redshift,mass,intial mass,metallicity, age) for stellar particles;" << endl;
  cout << " ..          #(Zgas_w, NHIgas_w) computed on galaxy-basis, for galaxies with id==is." << endl;
  cout << " ..          #(Np_sh, N_sim): number of particles with ids in the lightcone, number of particle with ids in the original simulation." << endl;
  cout << " ..          #(reddened flux in Nfilters) for stellar particles." << endl;
  cout << endl;

  string ncoord_path = rdir + "flux.dc." + snappl + "_" + conv(iplrestart, fINT) + ".txt";
  char out_buf_flux[65536];
  ofstream myfile2p;
  myfile2p.rdbuf()->pubsetbuf(out_buf_flux, sizeof(out_buf_flux));
  myfile2p.open(ncoord_path.c_str());
  if (!myfile2p.is_open()) {
    cerr << "Error: Could not open " << ncoord_path << " for writing." << endl;
    exit(2);
  }

  // Stream flux.df second pass
  ifstream ocf_pass2(filoutcat.c_str());
  if (!ocf_pass2.is_open()) {
    cerr << "Error: Could not re-open " << filoutcat << " for streaming output." << endl;
    exit(2);
  }
  char in_buf_df2[65536];
  ocf_pass2.rdbuf()->pubsetbuf(in_buf_df2, sizeof(in_buf_df2));

  size_t written_particles = 0;
  while (ocf_pass2 >> shid >> xi >> yi >> zri >> mi >> imi >> meti >> cmzsh >> ai >> NPTNG) {
    std::vector<double> temp_flux(Nfilters, 0.0);
    for (int cf = 0; cf < Nfilters; ++cf) {
      if (!(ocf_pass2 >> temp_flux[cf])) break;
    }
    ocf_pass2.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    size_t k = sh_to_idx[shid];

    if (read == 1) {
      myfile2g << shid << " " << xi << " " << yi << " " << zri << " "
               << mi << " " << imi << " " << meti << " " << ai << " "
               << Zm[k] << " " << NHIm[k] << " ";
      for (int f = 0; f < Nfilters; ++f) {
        myfile2g << temp_flux[f] << " ";
      }
      myfile2g << Npsh[k] << " " << static_cast<int>(NPTNG) << "\n";
    }

    int nptng_val = (k < NpshTNG_first.size()) ? NpshTNG_first[k] : static_cast<int>(NPTNG);
    myfile2p << shid << " " << xi << " " << yi << " " << zri << " "
             << mi << " " << imi << " " << meti << " " << ai << " "
             << Zm[k] << " " << NHIm[k] << " " << Npsh[k] << " " << nptng_val << " ";
    for (int f = 0; f < Nfilters; ++f) {
      myfile2p << static_cast<float>(temp_flux[f] * df[k][f]) << " ";
    }
    myfile2p << "\n";

    written_particles++;
  }
  ocf_pass2.close();

  if (read == 1) {
    myfile2g.close();
  }
  myfile2p.close();

  checkMemoryCeiling(memory_ceiling_gb);

  cout << endl;
  cout << " ... Flux file written ... " << endl;
  cout << endl;
  cout << written_particles << "        - (4) particles in final dust-corrected fov. " << endl;
  cout << endl;
  cout << "... But I can only show you the door. You're the one that has to walk through it ..." << endl;
  
  auto stop = high_resolution_clock::now(); 
  auto duration = duration_cast<microseconds>(stop - start);
  cout << "execution time in [h] " << 2.77778e-10 * duration.count() << endl;
  cout << " ------------------------------" << endl;

  return 0;
}
