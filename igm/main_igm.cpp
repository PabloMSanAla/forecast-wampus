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
#include <unordered_map>
#include <limits>
#include <sys/resource.h>
#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>
#endif
#define ARMA_DONT_USE_WRAPPER
#include <armadillo>
#include <CCfits/CCfits>

#include "readTNGParticle.h"
#include "SED.h"
#include "IGM_Ino14.h"
#include "functions.h"

/*****************************************************************************/
/*                                                                           */
/*             FORECAST - IGM-corrected fluxes calculations module           */
/*                                                                           */
/*  original dark matter-only code by cgiocoli@gmail.com                     */
/*  updated to its final form by flaminia.fortuni@inaf.it                    */
/*  optimized and RAM-aware streaming engine (2026)                          */
/*                                                                           */
/*****************************************************************************/

using namespace std;
using namespace std::chrono;
using namespace arma;
using namespace CCfits;

const double speedcunit = 2.99792458e+3;
const double mpM = 8.4089382e-58; // proton mass in Msun
const double mpg = 1.6726219e-24; // proton mass in g
const double mpc2tocm2 = 9.523e+42; // Mpc^2 to cm^2
const double h0 = 0.6774;

// Darwin/Linux RSS monitoring: returns peak RSS in MB
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

struct ParticleData {
  int shid;
  float x;
  float y;
  float zred;
  float m;
  float im;
  long double met;
  float age;
  float Zm;
  double NHIm;
  int npsh;
  int npshtng;
  vector<double> flux;
};

int main(int argc, char** argv) {
  auto start = high_resolution_clock::now();

  cout << "----------------------------------------------" << endl;
  cout << "-                                            -" << endl;
  cout << "::::IGM POST-PROCESSING for the LIGHTCONE::::" << endl;
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
      cout << "Usage: " << argv[0] << " <snapshot> <plane_number> [memory_ceiling_gb] [-ini <igm.ini>]" << endl;
      exit(0);
    } else {
      positional_args.push_back(arg);
    }
  }

  if (positional_args.size() < 2 || positional_args.size() > 3) {
    cerr << "Usage: " << argv[0] << " <snapshot> <plane_number> [memory_ceiling_gb] [-ini <igm.ini>]" << endl;
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

  // check if the file restart exists ... 
  std::string fileplstart = positional_args[1] + ".d";
  std::ifstream infileplstart(fileplstart.c_str());
  if (infileplstart.is_open()) {
    std::cout << "\n I will read the restart file >> " << fileplstart << std::endl;
    infileplstart >> iplrestart;
    std::cout << " iplrestart = " << iplrestart << "\n" << std::endl;
    infileplstart.close();
  }

  cout << "Initialize IGM module for plane number: " << iplrestart 
       << ", snapshot: " << sourceID 
       << ", memory ceiling: " << memory_ceiling_gb << " GB";
  if (!custom_ini.empty()) {
    cout << ", ini file: " << custom_ini;
  }
  cout << endl;

  checkMemoryCeiling(memory_ceiling_gb);

  // 2. Read Configuration from igm.ini
  double boxl;
  string filfilters, filsnaplist, filtimelist, idc;
  string pathsnap, bc03dir, dcpath, rdir;
  string model, imf;
  string planes_file_ini;

  readParameters(&boxl,
		 &filfilters, &filsnaplist, &filtimelist, &idc,
		 &pathsnap, &bc03dir, &dcpath, &rdir,
		 &model, &imf,
		 &planes_file_ini,
		 custom_ini);

  // 3. Resolve and validate planes_list.txt
  string fplane = "";
  const char* env_planes = std::getenv("FORECAST_PLANES_LIST");
  if (env_planes != nullptr && env_planes[0] != '\0') {
    fplane = env_planes;
  } else if (!planes_file_ini.empty()) {
    fplane = planes_file_ini;
  } else {
    fplane = "../lc/planes_list.txt";
  }

  ifstream oplane(fplane.c_str());
  if (!oplane.is_open()) {
    if (fplane != "lc/planes_list.txt" && fplane != "planes_list.txt") {
      oplane.open("lc/planes_list.txt");
      if (!oplane.is_open()) {
        oplane.open("planes_list.txt");
      }
    }
  }
  if (!oplane.is_open()) {
    cerr << "Error: planes list file " << fplane << " could not be found. Please check PLANES_LIST_FILE in igm.ini or FORECAST_PLANES_LIST env var." << endl;
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

  // 4. Load timelist, filters, cosmology, age list, and SSP tables
  ifstream timelist(filtimelist.c_str());
  if (!timelist.is_open()) {
    cerr << "Error: time list file " << filtimelist << " does not exist." << endl;
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

  // filters list in filters.dat
  ifstream listfilter(filfilters.c_str());
  if (!listfilter.is_open()) {
    cerr << "Error: filter list file " << filfilters << " does not exist." << endl;
    exit(2);
  }
  vector<string> nfilter;
  string filt_name;
  while (listfilter >> filt_name) {
    nfilter.push_back(filt_name);
  }
  listfilter.close();

  int Nfilters = (int)nfilter.size();
  std::vector<vector<long double>> fresp(Nfilters);
  std::vector<vector<long double>> fwaves(Nfilters);

  string filters_basedir = "";
  size_t last_slash = filfilters.find_last_of('/');
  if (last_slash != string::npos) {
    filters_basedir = filfilters.substr(0, last_slash + 1);
  }

  for (int N = 0; N < Nfilters; N++) {
    string filterin = filters_basedir + "filters/" + nfilter[N] + ".dat";
    ifstream filterlist(filterin.c_str());
    if (!filterlist.is_open()) {
      filterin = "../files/filters/" + nfilter[N] + ".dat";
      filterlist.open(filterin.c_str());
    }
    if (!filterlist.is_open()) {
      cerr << "Error: filter " << nfilter[N] << " is missing." << endl;
      exit(2);
    }
    long double sa, sb;
    while (filterlist >> sa >> sb) {
      fwaves[N].push_back(sa);
      fresp[N].push_back(sb);
    }
    filterlist.close();
  }

  // Precompute Filter Invariants (F7 Optimization)
  cout << "... Precomputing filter invariant cache ..." << endl;
  vector<FilterPrecomp> filter_cache(Nfilters);
  for (int N = 0; N < Nfilters; ++N) {
    filter_cache[N] = SED::precomputeFilter(nfilter[N], fwaves[N], fresp[N]);
  }

  // open age_bc03.txt / age_cb16.txt
  string filagelist = filters_basedir + "age_" + model + ".txt";
  ifstream agelist(filagelist.c_str());
  if (!agelist.is_open()) {
    filagelist = "../files/age_" + model + ".txt";
    agelist.open(filagelist.c_str());
  }
  if (!agelist.is_open()) {
    cerr << "Error: age list file " << filagelist << " does not exist." << endl;
    exit(2);
  }
  vector<double> nage;
  vector<long double> age_bc03;
  while (agelist >> buta >> butb) {
    nage.push_back(buta);
    age_bc03.push_back(butb);
  }
  agelist.close();

  // read Cosmology file: z, comoving_D, luminosity_D
  ifstream infiledc(idc.c_str());
  if (!infiledc.is_open()) {
    cerr << "Error: comoving distance file: " << idc << " does not exist." << endl;
    exit(2);
  }
  vector<double> zl, dl, dlum;
  double zi, dci, dli;
  while (infiledc >> zi >> dci >> dli) {
    zl.push_back(zi);
    dl.push_back(dci * speedcunit);
    dlum.push_back(dli * speedcunit);
  }
  infiledc.close();

  // SSP EXTRACTION
  std::vector<long double> waves;
  vector<vector<double>> time_grid;
  vector<vector<vector<double>>> full_table;
  if (model == "bc03") {
    waves.resize(1221);
    time_grid.resize(6);
    full_table.resize(6);
    for (int i = 0; i < 6; ++i) {
      readSSPTables(bc03dir, model, imf, i, waves, time_grid[i], full_table[i]);
    }
  } else if (model == "cb16") {
    waves.resize(13391);
    time_grid.resize(14);
    full_table.resize(14);
    for (int i = 0; i < 14; ++i) {
      readSSPTables(bc03dir, model, imf, i, waves, time_grid[i], full_table[i]);
    }
  }

  // 5. Read DC Output Catalog (dust-corrected fluxes)
  cout << "\n... Reading input file (flux.dc.snap_plane.txt) from previous module ..." << endl;
  cout << " .. content: #(ids,x,y,redshift,mass,intial mass,metallicity, age) for stellar particles;" << endl;
  cout << " ..          #(Zgas_w, NHIgas_w) computed on galaxy-basis, for galaxies with id==is." << endl;
  cout << " ..          #(Np_sh, N_sim): number of particles with ids in the lightcone, number of particle with ids in the original simulation." << endl;
  cout << " ..          #(reddened flux in Nfilters) for stellar particles.\n" << endl;

  if (!dcpath.empty() && dcpath.back() != '/') {
    dcpath += "/";
  }
  string filoutcat = dcpath + "flux.dc." + snappl + "_" + conv(iplrestart, fINT) + ".txt";
  ifstream ocf(filoutcat.c_str());
  if (!ocf.is_open()) {
    // Fallback: check output_baseline/dc/ if output_optimized/dc/ was specified
    string fallback_cat = filoutcat;
    size_t pos = fallback_cat.find("/output_optimized/dc/");
    if (pos != string::npos) {
      fallback_cat.replace(pos, 21, "/output_baseline/dc/");
      ocf.open(fallback_cat.c_str());
      if (ocf.is_open()) {
        filoutcat = fallback_cat;
      }
    }
  }
  if (!ocf.is_open()) {
    string local_cat = "flux.dc." + snappl + "_" + conv(iplrestart, fINT) + ".txt";
    ocf.open(local_cat.c_str());
    if (ocf.is_open()) {
      filoutcat = local_cat;
    }
  }
  if (!ocf.is_open()) {
    cerr << "Error: OUT_CAT file: " << filoutcat << " does not exist for this snapshot." << endl;
    exit(2);
  }

  // 6. Fast O(N) grouping by subhalo ID using hash map
  vector<ParticleData> particles;
  particles.reserve(100000);

  unordered_map<int, size_t> shid_to_subhalo_idx;
  vector<int> idsh;
  vector<vector<size_t>> subhalo_particles;
  vector<double> zrsh_sum;
  vector<vector<double>> fluxsh;

  int shid, npsh;
  float xi, yi, zri, mi, imi, ai, Zmi;
  long double meti, npshtng;
  double NHImi;

  while (ocf >> shid >> xi >> yi >> zri >> mi >> imi >> meti >> ai >> Zmi >> NHImi >> npsh >> npshtng) {
    ParticleData p;
    p.shid = shid;
    p.x = xi;
    p.y = yi;
    p.zred = zri;
    p.m = mi;
    p.im = imi;
    p.met = meti;
    p.age = ai;
    p.Zm = Zmi;
    p.NHIm = NHImi;
    p.npsh = npsh;
    p.npshtng = static_cast<int>(npshtng);
    p.flux.resize(Nfilters);

    for (int cf = 0; cf < Nfilters; ++cf) {
      if (!(ocf >> p.flux[cf])) {
        p.flux[cf] = 0.0;
      }
    }
    ocf.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    size_t p_idx = particles.size();
    particles.push_back(p);

    auto it = shid_to_subhalo_idx.find(shid);
    size_t k;
    if (it == shid_to_subhalo_idx.end()) {
      k = idsh.size();
      shid_to_subhalo_idx[shid] = k;
      idsh.push_back(shid);
      subhalo_particles.emplace_back();
      zrsh_sum.push_back(0.0);
      fluxsh.push_back(vector<double>(Nfilters, 0.0));
    } else {
      k = it->second;
    }

    subhalo_particles[k].push_back(p_idx);
    zrsh_sum[k] += zri;
    for (int cf = 0; cf < Nfilters; ++cf) {
      fluxsh[k][cf] += p.flux[cf];
    }
  }
  ocf.close();

  size_t num_subhalos = idsh.size();
  vector<int> Npsh(num_subhalos);
  vector<float> zrsh(num_subhalos);
  for (size_t k = 0; k < num_subhalos; ++k) {
    Npsh[k] = (int)subhalo_particles[k].size();
    zrsh[k] = static_cast<float>(zrsh_sum[k] / (double)Npsh[k]);
  }

  // printing some parameters for check
  SED sedy;
  IGM igmy;

  cout << "\n... Some parameters ..." << endl;
  cout << " (D_l, D_l2) = (" << blD << ", " << blD2 << ") cMpc; z between (" 
       << (sedy.getY(dl, zl, blD)) << ", " << (sedy.getY(dl, zl, blD2)) 
       << "); snapshot " << sourceID << ", plane " << n_pl << endl;
  cout << " filters: " << endl;
  for (size_t i = 0; i < nfilter.size(); i++) {
    cout << nfilter[i] << "  -  ";
  }
  cout << "\n.......................\n" << endl;

  int totPartxy4 = (int)particles.size();
  cout << "... Now assigning fluxes ...\n" << endl;
  cout << "N star particles: " << totPartxy4 << endl;
  cout << "N subhalos: " << num_subhalos << "\n" << endl;

  // 7. Calculate IGM attenuation factors per subhalo (O(N) total complexity)
  vector<vector<double>> df(num_subhalos, vector<double>(Nfilters, 1.0));
  vector<long double> met_bc03 { 0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05 };
  vector<long double> met_cb16 { 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.004, 0.006, 0.008, 0.010, 0.014, 0.017, 0.020, 0.030, 0.040 };


  size_t n_wave_pts = (model == "bc03") ? 1221 : 13391;
  vector<long double> spe;
  vector<long double> galSpe(n_wave_pts);
  vector<long double> wavesc(n_wave_pts);

  for (size_t k = 0; k < num_subhalos; ++k) {
    float zr = zrsh[k];
    std::fill(galSpe.begin(), galSpe.end(), 0.0);
    std::copy(waves.begin(), waves.end(), wavesc.begin());

    for (size_t p_idx : subhalo_particles[k]) {
      const auto& p = particles[p_idx];
      int age_inx = index_closest(age_bc03.begin(), age_bc03.end(), p.age);

      if (model == "bc03") {
        int met_inx = index_closest(met_bc03.begin(), met_bc03.end(), p.met);
        SEDbc03_interp_2spec(full_table[met_inx], time_grid[met_inx], age_inx, p.age, spe);
      } else if (model == "cb16") {
        int met_inx = index_closest(met_cb16.begin(), met_cb16.end(), p.met);
        SEDcb16_extract_spec(full_table[met_inx], time_grid[met_inx], age_inx, p.age, spe);
      }

      for (size_t i = 0; i < n_wave_pts; ++i) {
        galSpe[i] += spe[i] * p.im;
      }
    }

    // z evolution
    sedy.z_evol(zr, wavesc, galSpe, zl, dlum);

    // IGM attenuation
    igmy.igm_absorption(zr, wavesc, galSpe);

    // compute apparent magnitude and IGM attenuation factor in each filter
    for (int N = 0; N < Nfilters; ++N) {
      double m_ = sedy.compute_mab_fast(wavesc, galSpe, filter_cache[N]);
      double f_ = pow(10.0, (29.0 - (m_ + 48.6) / 2.5)); // uJy

      if (fluxsh[k][N] > f_) {
        df[k][N] = f_ / fluxsh[k][N];
      } else {
        df[k][N] = 1.0;
      }
    }

    if (k % 500 == 0) {
      checkMemoryCeiling(memory_ceiling_gb);
    }
  }

  checkMemoryCeiling(memory_ceiling_gb);

  cout << "\n... IGM effect on the particle flux ..." << endl;
  cout << "\n... Writing output file (flux.igm.snap_plane.txt) from this module ..." << endl;
  cout << " .. content: #(ids,x,y,redshift,mass,intial mass,metallicity, age) for stellar particles;" << endl;
  cout << " ..          #(Zgas_w, NHIgas_w) computed on galaxy-basis, for galaxies with id==is." << endl;
  cout << " ..          #(Np_sh, N_sim): number of particles with ids in the lightcone, number of particle with ids in the original simulation." << endl;
  cout << " ..          #(reddened+igm flux in Nfilters) for stellar particles.\n" << endl;

  // 8. Write output with 64KB I/O buffer (O(N) loop)
  if (!rdir.empty() && rdir.back() != '/') {
    rdir += "/";
  }
  string ncoord_path = rdir + "flux.igm." + snappl + "_" + conv(iplrestart, fINT) + ".txt";
  try {
    filesystem::create_directories(filesystem::path(ncoord_path).parent_path());
  } catch (...) {}

  ofstream myfile2p;
  vector<char> io_buffer(65536);
  myfile2p.rdbuf()->pubsetbuf(io_buffer.data(), io_buffer.size());
  myfile2p.open(ncoord_path.c_str());
  if (!myfile2p.is_open()) {
    cerr << "Error: Cannot open output file: " << ncoord_path << endl;
    exit(2);
  }

  for (size_t k = 0; k < num_subhalos; ++k) {
    for (size_t p_idx : subhalo_particles[k]) {
      const auto& p = particles[p_idx];
      myfile2p << p.shid << " " << p.x << " " << p.y << " " << p.zred << " "
               << p.m << " " << p.im << " " << p.met << " " << p.age << " "
               << p.Zm << " " << p.NHIm << " " << p.npsh << " " << p.npshtng << " ";
      for (int f = 0; f < Nfilters; ++f) {
        myfile2p << (p.flux[f] * df[k][f]) << " ";
      }
      myfile2p << "\n";
    }
  }
  myfile2p.close();

  cout << " IGM file written: " << ncoord_path << endl;
  cout << "\n... You are done, Neo ...\n" << endl;

  auto stop = high_resolution_clock::now(); 
  auto duration = duration_cast<microseconds>(stop - start);
  cout << "execution time in [h] " << 2.77778e-10 * duration.count() << endl;

  exit(0);
}
