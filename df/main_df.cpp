#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include <cmath>
#include <chrono> 
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <functional>
#include <stdio.h>     
#include <stdlib.h>     
#include <ctime>
#include <typeinfo>
#include <cassert>
#include <filesystem>
#include <sys/resource.h>
#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>
#endif
#ifdef COSMOLIB
#include <cosmo.h>
#endif
#include <H5Cpp.h>
#include <Eigen/Dense>
#include "readTNGParticle.h"
#define ARMA_DONT_USE_WRAPPER
#include <armadillo>
#include "SED.h"
#include "functions.h"

/*****************************************************************************
 *                                                                           
 *             FORECAST - dust-free fluxes calculations module               
 *                                                                           
 *  original dark matter-only code by cgiocoli@gmail.com                     
 *  updated to its final form by flaminia.fortuni@inaf.it                    
 *  - Updates 8/09/26: 
 *      Make RAM aware, add memory ceiling option, 
 *      optimize single threaded, and improve error handling.
 *                                                                           
*****************************************************************************/

using namespace std;

using namespace std;
using namespace arma;
using namespace std::chrono; 

const int bleft = 24;
const double speedcunit = 2.99792458e+3;
const double speedcunitas = 2.9979e+18; // AA/s
const double ergsa = 3.839e+33;
const double mpM = 8.4089382e-58; // proton mass in Msun
const double mpg = 1.6726219e-24; // proton mass in g

// ... TNG Cosmology
const double h0 = 0.6774;
const double om0 = 0.3089;
const double omL0 = 0.6911;

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

struct StarParticle {
  int shid;
  float x;
  float y;
  float zred;
  float m;
  float im;
  float age;
  float zstar;
  float cmzsh;
  long double met;
  int npsh;
};

int main(int argc, char** argv){
  auto start = high_resolution_clock::now();
  cout << "----------------------------------------------------------------------" << endl;
  cout << " " << endl; 
  cout << "   ------------------------------------------------------ " << endl;
  cout << "   -                                                    - " << endl;
  cout << "   -           2D Mapping Simulation Snapshot           - " << endl;
  cout << "   -                                                    - " << endl;
  cout << "   -                 Let there be light!                - " << endl;
  cout << "   ------------------------------------------------------ " << endl;

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
      cout << "Usage: " << argv[0] << " <snapshot> <plane_number> [memory_ceiling_gb] [-ini <df.ini>]" << endl;
      exit(0);
    } else {
      positional_args.push_back(arg);
    }
  }

  if (positional_args.size() < 2 || positional_args.size() > 3) {
    cerr << "Usage: " << argv[0] << " <snapshot> <plane_number> [memory_ceiling_gb] [-ini <df.ini>]" << endl;
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

  cout << "Initialize DF module for plane number: " << iplrestart 
       << ", snapshot: " << sourceID 
       << ", memory ceiling: " << memory_ceiling_gb << " GB";
  if (!custom_ini.empty()) {
    cout << ", ini file: " << custom_ini;
  }
  cout << endl;

  // Initial ceiling check
  checkMemoryCeiling(memory_ceiling_gb);

  // 2. Read Configuration from df.ini
  double boxl, zs;
  string filfilters, filredshiftlist, filsnaplist, filtimelist, idc;
  string pathsnap, lcpath, bc03dir, rdir; 
  string model, imf;
  string planes_file_ini;

  readParameters(&boxl, &zs,
                 &filfilters, &filredshiftlist, &filsnaplist, &filtimelist, &idc,
                 &pathsnap, &lcpath, &bc03dir, &rdir,
                 &model, &imf,
                 &planes_file_ini,
                 custom_ini);


  // 3. Resolve planes_list.txt (from ini file)
  string fplane = "";
  const char* env_planes = std::getenv("FORECAST_PLANES_LIST");
  if (env_planes != nullptr && env_planes[0] != '\0') {
    fplane = env_planes;
  } else if (!planes_file_ini.empty()) {
    fplane = planes_file_ini;
  }
  ifstream oplane;
  if (!fplane.empty()) {
    oplane.open(fplane.c_str());
  }
  if (!oplane.is_open()) {
    cerr << "Error: planes_list.txt could not be found. Please set PLANES_LIST_FILE in df.ini or FORECAST_PLANES_LIST env var." << endl;
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

  cout << "... Reading " << fplane << " file ... " << endl;
  cout << "( " << blD << ", " << blD2 << " ), " << snappl << ", " << zsim << endl;
  cout << " " << endl;

  // 4. Load Simulation Lookups & Cosmology Tables
  ifstream redlist(filredshiftlist.c_str());
  if (!redlist.is_open()) {
    cerr << "Error: redshift list file " << filredshiftlist << " could not be found. Please check FILE_REDSHIFT_LIST in df.ini." << endl;
    exit(2);
  }

  vector<int> tsnaplist;
  vector<double> dtsnaplist;
  vector<double> tredlist;
  int nmax = 1024;
  int buta;
  double butb, butc;
  while (redlist >> buta >> butb >> butc) {
    tsnaplist.push_back(buta);
    tredlist.push_back(butc);
    dtsnaplist.push_back(buta);
    if (buta > nmax) {
      cerr << "check nmax variable and increase it!" << endl;
      exit(1);
    }
  }
  redlist.close();

  // Load filter list
  cout << "... Reading filter list ..." << endl;
  ifstream listfilter(filfilters.c_str());
  if (!listfilter.is_open()) {
    cerr << "Error: filter list file " << filfilters << " could not be found. Please check FILTERS_FILE in df.ini." << endl;
    exit(2);
  }

  vector<string> nfilter;
  string filt_name;
  while (listfilter >> filt_name) {
    nfilter.push_back(filt_name);
  }
  listfilter.close();

  // Read filter files — derive filter directory from FILTERS_FILE parent path
  int Nfilters = (int)nfilter.size();
  vector<vector<long double>> fresp(Nfilters);
  vector<vector<long double>> fwaves(Nfilters);

  // Compute base directory from filfilters (e.g. "../files/filters.dat" -> "../files/")
  string filters_basedir = "";
  size_t last_slash = filfilters.find_last_of('/');
  if (last_slash != string::npos) {
    filters_basedir = filfilters.substr(0, last_slash + 1);
  }

  for (int N = 0; N < Nfilters; N++) {
    string filterin = filters_basedir + "filters/" + nfilter[N] + ".dat";
    ifstream filterlist(filterin.c_str());
    if (!filterlist.is_open()) {
      cerr << "Error: filter file " << filterin << " is missing. Check that FILTERS_FILE directory in df.ini contains a filters/ subdirectory." << endl;
      exit(2);
    }
    long double sa, sb;
    while (filterlist >> sa >> sb) {
      fwaves[N].push_back(sa);
      fresp[N].push_back(sb);
    }
    filterlist.close();
  }

  // Read timelist
  ifstream timelist(filtimelist.c_str());
  if (!timelist.is_open()) {
    cerr << "Error: time list file " << filtimelist << " could not be found. Please check FILE_TIMELIST in df.ini." << endl;
    exit(2);
  }

  vector<int> snap_tlist;
  vector<double> z_tlist;
  vector<double> age_tlist;
  while (timelist >> buta >> butb >> butc) {
    snap_tlist.push_back(buta);
    z_tlist.push_back(butb);
    age_tlist.push_back(butc);
  }
  timelist.close();

  // Open and read LCDM-comovingdistance
  ifstream infiledc(idc.c_str());
  if (!infiledc.is_open()) {
    cerr << "Error: comoving distance file " << idc << " could not be found. Please check PATH_AND_FILE_NAME_OF_COMOVING_DISTANCES in df.ini." << endl;
    exit(2);
  }

  vector<double> zl, dl, dlum;
  double zi_dc, dci, dli;
  while (infiledc >> zi_dc >> dci >> dli) {
    zl.push_back(zi_dc);
    dl.push_back(dci * speedcunit);
    dlum.push_back(dli * speedcunit);
  }
  infiledc.close();

  if (zl.empty() || zs > zl.back()) {
    cerr << "source redshift larger than the highest available redshift in comoving distance file" << endl;
    exit(1);
  }

  cout << "" << endl;
  cout << "-------------------------------------------" << endl;
  cout << " " << endl;
  cout << "... Starting to read TNG for snapshot " << snappl << " in filters " << endl;
  for (size_t i = 0; i < nfilter.size(); i++) cout << nfilter[i] << "  ||  ";
  cout << " ..." << endl;
  cout << " " << endl;

  // Stellar population model setup
  vector<long double> met_bc03 { 0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05 };
  vector<long double> met_cb16 { 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.004, 0.006, 0.008, 0.010, 0.014, 0.017, 0.020, 0.030, 0.040 };
  SED sedy;

  // Open age table — derive from FILTERS_FILE parent directory
  string filagelist = filters_basedir + "age_" + model + ".txt";
  ifstream agelist(filagelist.c_str());
  if (!agelist.is_open()) {
    cerr << "Error: age table file could not be found: " << filagelist << ". Check that FILTERS_FILE directory in df.ini contains age_" << model << ".txt." << endl;
    exit(2);
  }

  vector<double> nage;
  vector<long double> age_bc03;
  while (agelist >> buta >> butb) {
    nage.push_back(buta);
    age_bc03.push_back(butb);
  }
  agelist.close();

  // SSP EXTRACTION
  vector<long double> waves; // [AA]
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

  // 5. Precompute and Cache Filter Invariants (F7 Optimization)
  cout << "... Precomputing filter invariant cache ..." << endl;
  vector<FilterPrecomp> filter_cache(Nfilters);
  for (int N = 0; N < Nfilters; ++N) {
    filter_cache[N] = SED::precomputeFilter(nfilter[N], fwaves[N], fresp[N]);
  }

  // 6. Resolve Input Coordinates File from LC Module
  cout << ".. Reading input cone particle-based catalog from lc module.." << endl;
  cout << " .. content: #(ids,x,y,redshift,mass,initial mass,metallicity, -, age, -, CM of sh with ids, Nparticles of sh with ids in sim) for stellar particles;" << endl;
  cout << endl;

  string filoutcat = lcpath + "coords.lc." + snappl + "_" + conv(iplrestart, fINT) + ".txt";
  ifstream ocf(filoutcat.c_str());
  if (!ocf.is_open()) {
    // Fallback: check output_baseline/lc/ if output_optimized/lc/ was specified
    string fallback_cat = filoutcat;
    size_t pos = fallback_cat.find("/output_optimized/lc/");
    if (pos != string::npos) {
      fallback_cat.replace(pos, 21, "/output_baseline/lc/");
      ocf.open(fallback_cat.c_str());
      if (ocf.is_open()) {
        filoutcat = fallback_cat;
      }
    }
  }
  if (!ocf.is_open()) {
    string local_cat = "coords.lc." + snappl + "_" + conv(iplrestart, fINT) + ".txt";
    ocf.open(local_cat.c_str());
    if (ocf.is_open()) {
      filoutcat = local_cat;
    }
  }
  if (!ocf.is_open()) {
    cerr << "Error: OUT_CAT file: " << filoutcat << " does not exist." << endl;
    exit(2);
  }

  cout << "... Reading LC catalog: " << filoutcat << " ..." << endl;
  cout << " (D_l, D_l2) = (" << blD << ", " << blD2 << ") cMpc; z between (" 
       << (sedy.getY(dl, zl, blD)) << ", " << (sedy.getY(dl, zl, blD2)) 
       << "); snapshot " << sourceID << ", plane " << n_pl << endl;
  cout << endl;

  // 7. Prepare Buffered Output File
  string coord_path = rdir + "flux.df." + snappl + "_" + conv(iplrestart, fINT) + ".txt";
  ofstream myfile2;
  vector<char> io_buffer(65536);
  myfile2.rdbuf()->pubsetbuf(io_buffer.data(), io_buffer.size());

  // Ensure output directory exists if possible
  try {
    filesystem::create_directories(filesystem::path(coord_path).parent_path());
  } catch (...) {}

  myfile2.open(coord_path.c_str());
  if (!myfile2.is_open()) {
    cerr << "Error: Cannot open output file: " << coord_path << endl;
    exit(2);
  }

  // 8. Streaming Batch Processing (F6 RAM Ceiling & Bounded Footprint)
  const size_t BATCH_SIZE = 100000;
  vector<StarParticle> batch_particles;
  batch_particles.reserve(BATCH_SIZE);
  vector<double> batch_fluxes;
  batch_fluxes.resize(BATCH_SIZE * Nfilters);

  vector<long double> spe;
  vector<long double> wavesc;

  int shid;
  float xi, yi, zi, zri, mi, imi, ai, disttmp, sni, fiubb, cmzsh;
  long double meti, mabi, NPTNG;

  int totPartxy4 = 0;
  int first_subhalo_id = -1;
  int last_subhalo_id = -1;
  double xmin = 1e30, xmax = -1e30;
  double ymin = 1e30, ymax = -1e30;
  double zmin = 1e30, zmax = -1e30;
  double redmin = 1e30, redmax = -1e30;
  long double M_tot = 0.0;
  vector<long double> F_tot(Nfilters, 0.0);

  auto process_batch = [&]() {
    if (batch_particles.empty()) return;
    checkMemoryCeiling(memory_ceiling_gb);

    size_t count = batch_particles.size();
    for (size_t l = 0; l < count; ++l) {
      const auto& p = batch_particles[l];

      int age_inx = index_closest(age_bc03.begin(), age_bc03.end(), p.age);

      int met_inx = 0;
      if (model == "bc03") {
        wavesc.resize(1221);
        std::copy(waves.begin(), waves.end(), wavesc.begin());
        met_inx = index_closest(met_bc03.begin(), met_bc03.end(), p.met);
        SEDbc03_interp_2spec(full_table[met_inx], time_grid[met_inx], age_inx, p.age, spe);
      } else if (model == "cb16") {
        wavesc.resize(13391);
        std::copy(waves.begin(), waves.end(), wavesc.begin());
        met_inx = index_closest(met_cb16.begin(), met_cb16.end(), p.met);
        SEDcb16_extract_spec(full_table[met_inx], time_grid[met_inx], age_inx, p.age, spe);
      }

      // z evolution (in-place)
      sedy.z_evol(p.zred, wavesc, spe, zl, dlum);

      // Compute apparent magnitude and flux in each filter
      for (int N = 0; N < Nfilters; ++N) {
        double m_ = sedy.compute_mab_fast(wavesc, spe, filter_cache[N]);
        m_ = m_ - 2.5 * log10(p.im);
        double f_ = pow(10.0, (29.0 - (m_ + 48.6) / 2.5));
        batch_fluxes[l * Nfilters + N] = f_;
        F_tot[N] += f_;
      }
    }

    // Stream batch output directly to disk
    for (size_t l = 0; l < count; ++l) {
      const auto& p = batch_particles[l];
      myfile2 << p.shid << " " << p.x << " " << p.y << " " << p.zred << " "
              << p.m << " " << p.im << " " << p.met << " " << p.cmzsh << " "
              << p.age << " " << p.npsh << " ";
      for (int N = 0; N < Nfilters; ++N) {
        myfile2 << batch_fluxes[l * Nfilters + N] << " ";
      }
      myfile2 << "\n";
    }

    batch_particles.clear();
    checkMemoryCeiling(memory_ceiling_gb);
  };

  while (ocf >> shid >> xi >> yi >> zri >> mi >> imi >> meti >> fiubb >> ai >> zi >> cmzsh >> NPTNG) {
    if (shid > -1) {
      if (totPartxy4 == 0) {
        first_subhalo_id = shid;
      }
      last_subhalo_id = shid;

      xmin = std::min(xmin, (double)xi);
      xmax = std::max(xmax, (double)xi);
      ymin = std::min(ymin, (double)yi);
      ymax = std::max(ymax, (double)yi);
      zmin = std::min(zmin, (double)zi);
      zmax = std::max(zmax, (double)zi);
      redmin = std::min(redmin, (double)zri);
      redmax = std::max(redmax, (double)zri);
      M_tot += mi;
      totPartxy4++;

      batch_particles.push_back(StarParticle{
        shid, xi, yi, zri, mi, imi, ai, zi, cmzsh, meti, static_cast<int>(NPTNG)
      });

      if (batch_particles.size() >= BATCH_SIZE) {
        process_batch();
      }
    }
  }
  ocf.close();

  // Process remaining particles
  process_batch();
  myfile2.close();

  cout << totPartxy4 << "   type (4) - STAR particles selected in fov." << endl;
  if (totPartxy4 > 0) {
    cout << "first subhalo ID: " << first_subhalo_id << "  |  last subhalo ID: " << last_subhalo_id << endl;
  }
  cout << endl;
  cout << "      +++... Dust is not included in this run ...+++" << endl;
  cout << endl;

  if (totPartxy4 > 0) {
    cout << "... Min and max coordinates of star particles in fov ..." << endl;
    cout << "xmin = " << xmin << endl;
    cout << "xmax = " << xmax << endl;
    cout << "ymin = " << ymin << endl;
    cout << "ymax = " << ymax << endl;
    cout << "zmin = " << zmin << endl;
    cout << "zmax = " << zmax << endl;
    cout << "min redshift = " << redmin << endl;
    cout << "max redshift = " << redmax << endl;
    cout << endl;

    if (xmin < 0 || ymin < 0 || zmin < 0) {
      cerr << "  4 type check this!!! I will STOP here!!! " << endl;
      exit(1);
    }
  }

  cout << "****-fluxes computed-****" << endl;
  cout << endl;
  cout << ".. Flux file written: " << coord_path << " .." << endl;

  for (int N = 0; N < Nfilters; N++) {
    cout << " F_tot (total flux in fov [uJy]) for  " << nfilter[N] << " filter: " << F_tot[N] << endl;
  }
  cout << endl;
  cout << " M_tot (total mass in fov [M_sun]): " << M_tot << endl;
  cout << endl;
  cout << " N4_tot (total stellar particles in fov):  " << totPartxy4 << endl;
  cout << endl;

  auto stop = high_resolution_clock::now(); 
  auto duration = duration_cast<microseconds>(stop - start);
  cout << "execution time in [h] " << 2.77778e-10 * duration.count() << endl;
  cout << "----------------------------------------------------------------------" << endl;
  exit(0);
}
