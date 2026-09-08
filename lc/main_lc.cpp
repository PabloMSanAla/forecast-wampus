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
#include <bits/stdc++.h>
#include <typeinfo>
#include <cassert>
#include <dirent.h>
#include <sys/resource.h>
#include <unordered_map>
#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>
#endif
#include <H5Cpp.h>
#include <Eigen/Dense>
#include "readTNGParticle.h"
#define ARMA_DONT_USE_WRAPPER
#include <armadillo>
#include "functions.h"

/*****************************************************************************
 *                                                                           
 *             FORECAST - lightcone construction module                      
 *                                                                           
 *  original dark matter-only code by cgiocoli@gmail.com                     
 *  updated to its final form by flaminia.fortuni@inaf.it 
 *  Updates by Pablo M. Sanchez Alarcon - NASA Ames: 
 *  - 8/09/26: 
 *      Make RAM aware, add memory ceiling option, 
 *      optimize single threaded, and improve error handling.
 *                                                                           
*****************************************************************************/

using namespace std;
using namespace arma;
using namespace std::chrono; 

const int bleft = 24;
const double speedcunit = 2.99792458e+3;
const double speedcunitas = 2.9979e+18; //AA/s
const double ergsa = 3.839e+33;
const double mpM = 8.4089382e-58; //proton mass in Msun
const double mpg = 1.6726219e-24; //proton mass in g
const double h0 = 0.6774;

// Darwin-aware RSS monitoring: returns peak RSS in MB
double getPeakRSS_MB() {
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

struct PlaneEntry {
  int npl;
  double zpl;
  double blDpl;
  double blD2pl;
  int reppl;
  int blsnappl;
  double zpltrue;
};

int main(int argc, char** argv) {
  H5::Exception::dontPrint();
  auto start = high_resolution_clock::now();
  cout << "----------------------------------------------------------------------" << endl;
  cout << " " << endl; 
  cout << "   ------------------------------------------------------ " << endl;
  cout << "   -                                                    - " << endl;
  cout << "   -           2D Mapping Simulation Snapshot           - " << endl;
  cout << "   -                                                    - " << endl;
  cout << "   -               building the light-cone              - " << endl;
  cout << "   ------------------------------------------------------ " << endl;

  // 1. CLI Parsing & Validation
  int target_snap = -1;
  int target_plane = -1;
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
      cout << "Usage: " << argv[0] << " <snapshot> <plane_number> [memory_ceiling_gb] [-ini <lc.ini>]" << endl;
      exit(0);
    } else {
      positional_args.push_back(arg);
    }
  }

  if (positional_args.size() < 2 || positional_args.size() > 3) {
    cerr << "Usage: " << argv[0] << " <snapshot> <plane_number> [memory_ceiling_gb] [-ini <lc.ini>]" << endl;
    exit(1);
  }

  try {
    target_snap = std::stoi(positional_args[0]);
    target_plane = std::stoi(positional_args[1]);
  } catch (...) {
    cerr << "Error: Both snapshot and plane number must be valid integers." << endl;
    exit(1);
  }

  if (target_snap < 0 || target_plane < 0) {
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

  cout << "Requested snapshot: " << target_snap 
       << ", plane: " << target_plane 
       << ", memory ceiling: " << memory_ceiling_gb << " GB";
  if (!custom_ini.empty()) {
    cout << ", ini file: " << custom_ini;
  }
  cout << endl;

  // Check initial memory against ceiling
  checkMemoryCeiling(memory_ceiling_gb);

  // 2. Read Input Configuration (lc.ini)
  double boxl, zs, Ds, fov, res;
  string filredshiftlist, filsnaplist, filtimelist, idc;
  string pathsnap, rdir;
  long seedcenter, seedface, seedsign;
  string sim;
  string planes_file_ini;

  readParameters(&boxl, &zs, &fov, &res,
                 &filredshiftlist, &filsnaplist, &filtimelist, &idc,
                 &pathsnap, &rdir,
                 &seedcenter, &seedface, &seedsign, &sim,
                 &planes_file_ini,
                 custom_ini);


  int truenpix = int(fov * 3600.0 / res);
  int bufferpix = int(ceil((truenpix + 1) * 20 / 14142));
  int npix = truenpix + bufferpix;

  cout << "N. pixels: " << truenpix << "; buffer pixels: " << bufferpix << endl;
  cout << endl;

  // 3. Resolve and Validate planes_list.txt (from ini file)
  string planes_file = "";
  const char* env_planes = std::getenv("FORECAST_PLANES_LIST");
  if (env_planes != nullptr && env_planes[0] != '\0') {
    planes_file = env_planes;
  } else if (!planes_file_ini.empty()) {
    planes_file = planes_file_ini;
  }
  ifstream infile_planes;
  if (!planes_file.empty()) {
    infile_planes.open(planes_file);
  }
  if (!infile_planes.is_open()) {
    cerr << "Error: planes_list.txt could not be found. Please set PLANES_LIST_FILE in lc.ini or FORECAST_PLANES_LIST env var." << endl;
    exit(2);
  }

  vector<PlaneEntry> all_planes;
  int matched_plane_idx = -1;
  int npl_in, reppl_in, blsnappl_in;
  double zpl_in, blDpl_in, blD2pl_in, zpltrue_in;

  while (infile_planes >> npl_in >> zpl_in >> blDpl_in >> blD2pl_in >> reppl_in >> blsnappl_in >> zpltrue_in) {
    PlaneEntry pe{npl_in, zpl_in, blDpl_in, blD2pl_in, reppl_in, blsnappl_in, zpltrue_in};
    int current_idx = (int)all_planes.size();
    if (blsnappl_in == target_snap && (reppl_in == target_plane || npl_in - 1 == target_plane)) {
      matched_plane_idx = current_idx;
    }
    all_planes.push_back(pe);
  }
  infile_planes.close();

  if (matched_plane_idx == -1) {
    cerr << "Error: Snapshot " << target_snap << " and plane " << target_plane 
         << " do not match any entry in " << planes_file << "." << endl;
    exit(1);
  }

  const PlaneEntry& selected_plane = all_planes[matched_plane_idx];
  int nsnap = matched_plane_idx;
  int rcase = selected_plane.reppl;
  int sourceID = selected_plane.blsnappl;
  string snappl = std::to_string(sourceID);
  double blD_nsnap = selected_plane.blDpl;
  double blD2_nsnap = selected_plane.blD2pl;

  cout << "Matched Plane: Row " << selected_plane.npl 
       << ", Replica " << selected_plane.reppl 
       << ", Snapshot " << sourceID 
       << ", Near: " << blD_nsnap << " cMpc/h, Far: " << blD2_nsnap << " cMpc/h" << endl;

  // 4. Load Simulation Lookups & Cosmology Tables
  ifstream redlist(filredshiftlist.c_str());
  vector<int> tsnaplist;
  vector<double> dtsnaplist;
  vector<double> tredlist;
  int nmax = 1024;
  vector<double> snapToRedshift(nmax, -1.0);
  if (redlist.is_open()) {
    int buta;
    double butb, butc;
    while (redlist >> buta >> butb >> butc) {
      tsnaplist.push_back(buta);
      tredlist.push_back(butc);
      dtsnaplist.push_back(buta);
      if (buta < nmax) snapToRedshift[buta] = butc;
    }
    redlist.close();
  } else {
    cerr << "Error: redshift list file " << filredshiftlist << " does not exist." << endl;
    exit(2);
  }

  ifstream timelist(filtimelist.c_str());
  vector<double> z_tlist;
  vector<double> age_tlist;
  if (timelist.is_open()) {
    int buta;
    double butb, butc;
    while (timelist >> buta >> butb >> butc) {
      z_tlist.push_back(butb);
      age_tlist.push_back(butc);
    }
    timelist.close();
  } else {
    cerr << "Error: time list file " << filtimelist << " does not exist." << endl;
    exit(2);
  }

  ifstream snaplist(filsnaplist.c_str());
  vector<int> lsnap;
  vector<double> lred;
  if (snaplist.is_open()) {
    int s;
    while (snaplist >> s) {
      double zn = getY(dtsnaplist, tredlist, double(s));
      if (zn <= zs) {
        lsnap.push_back(s);
        lred.push_back(zn);
      }
    }
    snaplist.close();
  } else {
    cerr << "Error: snaplist file " << filsnaplist << " does not exist." << endl;
    exit(2);
  }

  cout << "Opening path for snapshots >> " << pathsnap << endl;
  cout << "Looking for comoving distance file >> " << idc << endl;

  ifstream infiledc(idc.c_str());
  if (!infiledc.is_open()) {
    cerr << "Error: comoving distance file " << idc << " does not exist. Please check PATH_AND_FILE_NAME_OF_COMOVING_DISTANCES in lc.ini." << endl;
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

  if (zs > zl.back()) {
    cerr << "Error: source redshift larger than comoving distance file maximum." << endl;
    exit(1);
  }

  Ds = getY(zl, dl, zs);

  // Compute exact in-memory cosmology plane tables for full double precision
  int nsnaps_init = lsnap.size();
  int nreplications = int(Ds / boxl) + 1;

  vector<int> replication, fromsnap;
  vector<double> lD, lD2;

  for (int j = 0; j < nreplications; j++) {
    for (int i = 0; i < nsnaps_init; i++) {
      double ldbut = getY(zl, dl, lred[i]);
      if (ldbut >= j * boxl && ldbut <= (j + 1.) * boxl) {
        replication.push_back(j);
        fromsnap.push_back(lsnap[i]);
        lD.push_back(ldbut);
      }
    }
  }

  for (size_t i = 0; i < lD.size(); i++) {
    if (i < (lD.size() - 1)) lD2.push_back(lD[i + 1]);
    else lD2.push_back(Ds);
  }
  // To safely allow lD[i+1] assignment when i+1 == lD.size()
  lD.push_back(Ds);

  for (size_t i = 0; i < lD2.size(); i++) {
    for (int k = 1; k <= 512; k++) {
      if (lD[i] < double(k) * boxl && lD2[i] > double(k) * boxl) {
        lD[i + 1] = lD2[i];
        lD2[i] = double(k) * boxl;
      }
    }
  }
  lD.pop_back();

  vector<double> zsimlens(nsnaps_init);
  for (int i = 0; i < nsnaps_init; i++) {
    if (i < nsnaps_init - 1) {
      if (lD[i + 1] - lD2[i] > boxl * 1e-9) {
        fromsnap[i] = -fromsnap[i];
      }
    }
    double dlbut = (lD[i] + lD2[i]) * 0.5;
    zsimlens[i] = getY(dl, zl, dlbut);
  }

  vector<double> Bfromsnap, BlD, BlD2, Bzsimlens, Blred;
  vector<int> Breplication, Blsnap;

  for (int i = 0; i < nsnaps_init; i++) {
    Bfromsnap.push_back(fabs(fromsnap[i]));
    BlD.push_back(lD[i]);
    BlD2.push_back(lD2[i]);
    Bzsimlens.push_back(zsimlens[i]);
    Breplication.push_back(replication[i]);
    Blsnap.push_back(lsnap[i]);
    Blred.push_back(lred[i]);
    if (fromsnap[i] < 0) {
      Bfromsnap.push_back(-fromsnap[i]);
      BlD.push_back(lD2[i]);
      BlD2.push_back(lD[i + 1]);
      double dlbut = (lD[i + 1] + lD2[i]) * 0.5;
      Bzsimlens.push_back(getY(dl, zl, dlbut));
      Breplication.push_back(replication[i + 1]);
      Blsnap.push_back(lsnap[i]);
      Blred.push_back(lred[i]);
    }
  }

  // Add sentinel element to BlD and BlD2 to allow safe [i+1] reads
  BlD.push_back(BlD.back());
  BlD2.push_back(BlD2.back());

  vector<double> bbfromsnap, bblD, bblD2, bbzsimlens, bblred;
  vector<int> bbreplication, bblsnap;

  for (size_t i = 0; i < Blred.size(); i++) {
    float delta = BlD2[i] - BlD[i];
    float delta1 = BlD2[i + 1] - BlD[i + 1];
    int n = static_cast<int>(std::ceil(delta / boxl));
    if (delta > boxl) {
      double add = delta / static_cast<double>(n);
      for (int k = 1; k <= n; k++) {
        bblD.push_back(BlD[i] + (k - 1) * add);
        bblD2.push_back(BlD[i] + k * add);
        double dlbut = BlD[i] + (2 * k - 1) * 0.5 * add;
        bbzsimlens.push_back(getY(dl, zl, dlbut));
        bbreplication.push_back((i > 0 ? Breplication[i - 1] : 0) + k);
        bblsnap.push_back(Blsnap[i]);
        bblred.push_back(Blred[i]);
        bbfromsnap.push_back(Bfromsnap[i]);
      }
    } else {
      bblD.push_back(BlD[i]);
      bblD2.push_back(BlD[i] + delta);
      double dlbut = BlD[i] + 0.5 * delta;
      bbzsimlens.push_back(getY(dl, zl, dlbut));
      bbreplication.push_back((i > 0 ? Breplication[i - 1] : 0) + 1);
      bblsnap.push_back(Blsnap[i]);
      bblred.push_back(Blred[i]);
      bbfromsnap.push_back(Bfromsnap[i]);
    }

    if (i + 1 < Blred.size() && (delta + delta1) / 2. <= boxl && Bfromsnap[i] == Bfromsnap[i + 1]) {
      bblD.push_back(BlD[i]);
      bblD2.push_back((BlD[i] + BlD2[i + 1]) / 2.);
      double dlbut = (BlD[i] + (BlD2[i + 1] + BlD[i]) / 2.) / 2.;
      bbzsimlens.push_back(getY(dl, zl, dlbut));
      bbreplication.push_back(Breplication[i]);
      bblsnap.push_back(Blsnap[i]);
      bblred.push_back(Blred[i]);
      bbfromsnap.push_back(Bfromsnap[i]);

      bblD.push_back((BlD2[i + 1] + BlD[i]) / 2.);
      bblD2.push_back(BlD2[i + 1]);
      double dlbut2 = (BlD2[i + 1] + (BlD2[i + 1] + BlD[i]) / 2.) / 2.;
      bbzsimlens.push_back(getY(dl, zl, dlbut2));
      bbreplication.push_back(Breplication[i] + 1);
      bblsnap.push_back(Blsnap[i]);
      bblred.push_back(Blred[i]);
      bbfromsnap.push_back(Bfromsnap[i]);
      i = i + 1;
    }
  }

  // Sentinel for bbl
  bblD.push_back(bblD.back());
  bblD2.push_back(bblD2.back());
  bbfromsnap.push_back(bbfromsnap.back());

  vector<double> bbbfromsnap, bbblD, bbblD2, bbbzsimlens, bbblred;
  vector<int> bbbreplication, bbblsnap;
  for (size_t i = 0; i < bblred.size(); i++) {
    float delta = bblD2[i] - bblD[i];
    float delta1 = bblD2[i + 1] - bblD[i + 1];
    if (i + 1 < bblred.size() && delta + delta1 <= boxl && bbfromsnap[i] == bbfromsnap[i + 1]) {
      bbbfromsnap.push_back(bbfromsnap[i]);
      bbblD.push_back(bblD[i]);
      bbblD2.push_back(bblD2[i + 1]);
      double dlbut = (bblD[i] + bblD2[i + 1]) / 2.;
      bbbzsimlens.push_back(getY(dl, zl, dlbut));
      bbbreplication.push_back(bbreplication[i] + 1);
      bbblsnap.push_back(bblsnap[i]);
      bbblred.push_back(bblred[i]);
      i = i + 1;
    } else {
      bbbfromsnap.push_back(bbfromsnap[i]);
      bbblD.push_back(bblD[i]);
      bbblD2.push_back(bblD2[i]);
      bbbzsimlens.push_back(bbzsimlens[i]);
      bbbreplication.push_back(bbreplication[i]);
      bbblsnap.push_back(bblsnap[i]);
      bbblred.push_back(bblred[i]);
    }
  }

  // Sentinel for bbbl
  bbblD.push_back(bbblD.back());
  bbblD2.push_back(bbblD2.back());
  bbbfromsnap.push_back(bbbfromsnap.back());

  vector<double> bfromsnap, blD, blD2, bzsimlens, blred;
  vector<int> breplication, blsnap;
  for (size_t i = 0; i < bbblred.size(); i++) {
    float delta = bbblD2[i] - bbblD[i];
    float delta1 = bbblD2[i + 1] - bbblD[i + 1];
    if (i + 1 < bbblred.size() && delta + delta1 <= boxl && bbbfromsnap[i] == bbbfromsnap[i + 1]) {
      bfromsnap.push_back(bbbfromsnap[i]);
      blD.push_back(bbblD[i]);
      blD2.push_back(bbblD2[i + 1]);
      double dlbut = (bbblD[i] + bbblD2[i + 1]) / 2.;
      bzsimlens.push_back(getY(dl, zl, dlbut));
      breplication.push_back(bbbreplication[i] + 1);
      blsnap.push_back(bbblsnap[i]);
      blred.push_back(bbblred[i]);
      i = i + 1;
    } else {
      bfromsnap.push_back(bbbfromsnap[i]);
      blD.push_back(bbblD[i]);
      blD2.push_back(bbblD2[i]);
      bzsimlens.push_back(bbbzsimlens[i]);
      breplication.push_back(bbbreplication[i]);
      blsnap.push_back(bbblsnap[i]);
      blred.push_back(bbblred[i]);
    }
  }

  vector<int> er_inx;
  for (size_t i = 0; i < blD.size() - 1; i++) {
    if (blD[i] == blD[i + 1] && blD2[i] != blD2[i + 1]) {
      er_inx.push_back(i);
    }
  }
  for (auto it = er_inx.rbegin(); it != er_inx.rend(); ++it) {
    bfromsnap.erase(bfromsnap.begin() + *it);
    blD.erase(blD.begin() + *it);
    blD2.erase(blD2.begin() + *it);
    bzsimlens.erase(bzsimlens.begin() + *it);
    breplication.erase(breplication.begin() + *it);
    blsnap.erase(blsnap.begin() + *it);
    blred.erase(blred.begin() + *it);
  }

  breplication.clear();
  int nsnaps = bfromsnap.size();
  for (int i = 0; i < nsnaps; i++) {
    breplication.push_back(i);
  }

  if (matched_plane_idx < (int)blD.size()) {
    blD_nsnap = blD[matched_plane_idx];
    blD2_nsnap = blD2[matched_plane_idx];
  }

  double truefov = fov;
  double fovradiants = fov / 180.0 * M_PI;

  if (fovradiants * Ds > boxl) {
    cerr << "Error: field of view too large for box size." << endl;
    exit(1);
  }

  fov = truefov * double(npix) / double(truenpix);
  fovradiants = fov / 180.0 * M_PI;

  // 5. Randomization of Box Realizations
  int nrandom = all_planes.back().reppl + 1;
  vector<double> x0(nrandom), y0(nrandom), z0(nrandom);
  vector<int> face(nrandom);
  vector<int> sgnX(nrandom), sgnY(nrandom), sgnZ(nrandom);

  for (int i = 0; i < nrandom; i++) {
    if (seedcenter > 0) {
      srand(seedcenter + i * 13);
      x0[i] = rand() / float(RAND_MAX);
      y0[i] = rand() / float(RAND_MAX);
      z0[i] = rand() / float(RAND_MAX);
    } else {
      x0[i] = 0.0;
      y0[i] = 0.0;
      z0[i] = 0.0;
    }
    face[i] = 7;
    if (seedface > 0) {
      srand(seedface + i * 5);
      while (face[i] > 6 || face[i] < 1)
        face[i] = int(1 + rand() / float(RAND_MAX) * 5.0 + 0.5);
    } else {
      face[i] = 1;
    }
    sgnX[i] = 2;
    if (seedsign > 0) {
      srand(seedsign + i * 8);
      while (sgnX[i] > 1 || sgnX[i] < 0) sgnX[i] = int(rand() / float(RAND_MAX) + 0.5);
      sgnY[i] = 2;
      while (sgnY[i] > 1 || sgnY[i] < 0) sgnY[i] = int(rand() / float(RAND_MAX) + 0.5);
      sgnZ[i] = 2;
      while (sgnZ[i] > 1 || sgnZ[i] < 0) sgnZ[i] = int(rand() / float(RAND_MAX) + 0.5);
      if (sgnX[i] == 0) sgnX[i] = -1;
      if (sgnY[i] == 0) sgnY[i] = -1;
      if (sgnZ[i] == 0) sgnZ[i] = -1;
    } else {
      sgnX[i] = 1;
      sgnY[i] = 1;
      sgnZ[i] = 1;
    }
  }

  // 6. Initialize Simulation Data & Group Catalog
  cout << endl;
  cout << "----------------------------------------------------------------------" << endl;
  cout << "... Starting to read TNG for snapshot " << snappl << "..." << endl;
  cout << endl;

  string check_snap_file = pathsnap + "/snapdir_0" + snappl + "/snap_0" + snappl + ".0.hdf5";
  ifstream test_snap(check_snap_file);
  if (!test_snap.is_open()) {
    cerr << "Error: Missing snapshot data file: " << check_snap_file << endl;
    exit(2);
  }
  test_snap.close();

  vector<int> gcLenTypeS;
  vector<int> gcOffsetsTypeS;
  vector<int> ids;
  vector<float> xs, ys, zreds, ms4, ims4, ages4, zstar, orgZ;
  vector<long double> mets4;
  vector<double> fiub, xtmp, ytmp;
  vector<int> countSH4;
  double xmin = 1e30, xmax = -1e30;
  double ymin = 1e30, ymax = -1e30;
  double zmin = 1e30, zmax = -1e30;
  double zstarmin = 1e30, zstarmax = -1e30;
  long long n4_total = 0;

  try {
    readTNGParticle tngParticle;
    tngParticle.Initialize(pathsnap, sourceID);
    tngParticle.readHeader(0);

    double bs = tngParticle.getBoxSize();
    double om0 = tngParticle.getOmegaZero();
    double omL0 = tngParticle.getOmegaLambda();
    double time = tngParticle.getTime();
    double zsim = tngParticle.getRedshift();
    double dlsim = getY(zl, dl, zsim);
    vector<double> mass = tngParticle.getMassTable();
    vector<int> npart = tngParticle.getNumPartTotal();

    string workdir_sn = pathsnap + "/snapdir_0" + snappl + "/";
    string workdir_g = pathsnap + "/groups_0" + snappl + "/";
    int nf_sn = countHDF5Files(workdir_sn, "hdf5");
    int nf_g = countHDF5Files(workdir_g, "hdf5");

    for (int i = 0; i < nf_g; i++) {
      tngParticle.readFof(i);
      vector<int> sLTS = tngParticle.getStarsLenType();
      gcLenTypeS.insert(gcLenTypeS.end(), sLTS.begin(), sLTS.end());
    }
    tngParticle.readOffset();
    gcOffsetsTypeS = tngParticle.getStarsByType();

    int num_sh = (int)gcLenTypeS.size();
    vector<int> diff(num_sh);
    vector<int> gcOffsetsMax(num_sh);
    for (int s = 0; s < num_sh; s++) {
      diff[s] = gcOffsetsTypeS[s] - 1;
      gcOffsetsMax[s] = gcOffsetsTypeS[s] + gcLenTypeS[s] - 1;
    }

    // Hoisted invariants (F3 optimization)
    float convm = float(1.e10 / h0);
    float t0 = getY(z_tlist, age_tlist, zsim);

    countSH4.assign(num_sh, 0);

    int cur_sub = 0;
    int global_id = 0;

    cout << "Ingesting particles file-by-file across " << nf_sn << " chunk files..." << endl;

    for (int f = 0; f < nf_sn; f++) {
      tngParticle.readStars(f);
      const vector<double>& metal = tngParticle.getMetallicity();
      const vector<double>& inMass = tngParticle.getInitialMass();
      const vector<double>& sTime = tngParticle.getStellarFormationTime();
      const vector<double>& Mass = tngParticle.getMasses();
      const vector<double>& X_ = tngParticle.getX();
      const vector<double>& Y_ = tngParticle.getY();
      const vector<double>& Z_ = tngParticle.getZ();

      int npart_file = (int)metal.size();

      for (int p = 0; p < npart_file; p++) {
        int pid = global_id + p;
        float ftime = sTime[p];
        if (ftime <= 0.0f) continue;

        // Track subhalo particle count across full simulation in O(1) amortized
        while (cur_sub + 1 < num_sh && pid > diff[cur_sub + 1]) {
          cur_sub++;
        }
        int sh_id = -1;
        if (pid >= gcOffsetsTypeS[0] && cur_sub < num_sh && pid <= gcOffsetsMax[cur_sub]) {
          sh_id = cur_sub;
        }
        if (sh_id > -1) {
          countSH4[sh_id]++;
        }

        // Coordinate manipulation in the simulation box
        float xb = sgnX[rcase] * (float(X_[p]) / bs);
        float yb = sgnY[rcase] * (float(Y_[p]) / bs);
        float zb = sgnZ[rcase] * (float(Z_[p]) / bs);
        float zr = zb;
        float orgz = float(Z_[p]);

        if (xb > 1.0f) xb -= 1.0f;
        if (yb > 1.0f) yb -= 1.0f;
        if (zb > 1.0f) zb -= 1.0f;
        if (zr > 1.0f) zr -= 1.0f;
        if (xb < 0.0f) xb += 1.0f;
        if (yb < 0.0f) yb += 1.0f;
        if (zb < 0.0f) zb += 1.0f;
        if (zr < 0.0f) zr += 1.0f;

        float x, y, z, zred;
        switch (face[rcase]) {
          case 1: x = xb; y = yb; z = zb; zred = zr; break;
          case 2: x = xb; y = zb; z = yb; zred = yb; break;
          case 3: x = yb; y = zb; z = xb; zred = xb; break;
          case 4: x = yb; y = xb; z = zb; zred = zr; break;
          case 5: x = zb; y = xb; z = yb; zred = yb; break;
          case 6: x = zb; y = yb; z = xb; zred = xb; break;
          default: x = xb; y = yb; z = zb; zred = zr; break;
        }

        x -= x0[rcase];
        y -= y0[rcase];
        z -= z0[rcase];
        zred -= z0[rcase];

        if (x > 1.0f) x -= 1.0f;
        if (y > 1.0f) y -= 1.0f;
        if (z > 1.0f) z -= 1.0f;
        if (zred > 1.0f) zred -= 1.0f;
        if (x < 0.0f) x += 1.0f;
        if (y < 0.0f) y += 1.0f;
        if (z < 0.0f) z += 1.0f;
        if (zred < 0.0f) zred += 1.0f;

        float zzs_val = z;
        float zz4_val = z + float(rcase);
        xmin = std::min(xmin, (double)x);
        xmax = std::max(xmax, (double)x);
        ymin = std::min(ymin, (double)y);
        ymax = std::max(ymax, (double)y);
        zmin = std::min(zmin, (double)zz4_val);
        zmax = std::max(zmax, (double)zz4_val);
        zstarmin = std::min(zstarmin, (double)zzs_val);
        zstarmax = std::max(zstarmax, (double)zzs_val);
        n4_total++;

        // Field of view filtering (inline multiplication replacing pow)
        double Ztmp = (zzs_val * (bs / 1.e+3) + blD_nsnap) / (bs / 1.e+3);
        double dx = x - 0.5;
        double dy = y - 0.5;
        double di = sqrt(dx * dx + dy * dy + Ztmp * Ztmp) * (bs / 1.e+3);

        if (di >= blD_nsnap && di < blD2_nsnap) {
          double rai, deci, dd;
          getPolar(dx, dy, Ztmp, &rai, &deci, &dd);
          if (fabs(rai) <= fovradiants * 0.5 && fabs(deci) <= fovradiants * 0.5) {
            double fovinunitbox = fovradiants * di / (bs / 1.e+3);
            float zf = 1.0 / double(ftime) - 1.0;
            float tf = getY(z_tlist, age_tlist, zf);
            ids.push_back(pid);
            fiub.push_back(fovinunitbox);
            xs.push_back((x - 0.5) / fovinunitbox + 0.5);
            ys.push_back((y - 0.5) / fovinunitbox + 0.5);
            zstar.push_back(zzs_val / fovinunitbox);
            zreds.push_back(getY(dl, zl, Ztmp * (bs / 1.e+3)));
            ms4.push_back(float(Mass[p]) * convm);
            ims4.push_back(float(inMass[p]) * convm);
            mets4.push_back((long double)float(metal[p]));
            ages4.push_back(t0 - tf);
            xtmp.push_back(x * (bs / 1.e+3));
            ytmp.push_back(y * (bs / 1.e+3));
            orgZ.push_back(orgz);
          }
        }
      }

      global_id += npart_file;

      // Continuous Peak RSS check against memory ceiling
      checkMemoryCeiling(memory_ceiling_gb);
    }
  } catch (const H5::Exception& e) {
    cerr << "Error: Missing or inaccessible HDF5 dataset/file: " << e.getDetailMsg() << endl;
    exit(2);
  } catch (const std::exception& e) {
    cerr << "Error: Failed to process snapshot data: " << e.what() << endl;
    exit(2);
  }

  cout << n4_total << "   type (4) - STAR particles re-arranged in the snapshot." << endl;
  cout << endl;
  cout << " n4 particles " << endl;
  cout << "xmin = " << xmin << "\nxmax = " << xmax << endl;
  cout << "ymin = " << ymin << "\nymax = " << ymax << endl;
  cout << "zmin = " << zstarmin << "\nzmax = " << zstarmax << endl;
  cout << "boxmin = " << zmin << "\nboxmax = " << zmax << endl;
  cout << endl;
  cout << " ... selecting only particles in fov ..." << endl;
  cout << endl;

  cout << " ... Coordinates & redshift limits ..." << endl;
  if (!xtmp.empty() && !ytmp.empty() && !zreds.empty()) {
    cout << "xmin, xmax in Mpc/h " << double(*min_element(xtmp.begin(), xtmp.end())) << ", " << double(*max_element(xtmp.begin(), xtmp.end())) << endl;
    cout << "ymin, ymax in Mpc/h " << double(*min_element(ytmp.begin(), ytmp.end())) << ", " << double(*max_element(ytmp.begin(), ytmp.end())) << endl;
    cout << "zmin, zmax in Mpc/h " << double(*min_element(zreds.begin(), zreds.end())) << ", " << double(*max_element(zreds.begin(), zreds.end())) << endl;
    cout << endl;
  }

  // 8. Subhalo Mapping & Center of Mass Calculation (F3 single-core optimization)
  vector<int> idshs = inverseMap_sh_idv3(gcLenTypeS, gcOffsetsTypeS, ids, true);

  vector<int> idshtmp;
  for (size_t i = 0; i < idshs.size(); i++) {
    if (idshs[i] > -1)
      idshtmp.push_back(idshs[i]);
  }
  vector<int> idsh;
  std::unique_copy(idshtmp.begin(), idshtmp.end(), std::back_inserter(idsh));
  idshtmp.clear();
  idshtmp.shrink_to_fit();

  // Fast O(L) group mapping: pre-index particle locations per subhalo
  unordered_map<int, vector<int>> sh_to_indices;
  sh_to_indices.reserve(idsh.size());
  for (int l = 0; l < (int)idshs.size(); l++) {
    if (idshs[l] > -1) {
      sh_to_indices[idshs[l]].push_back(l);
    }
  }

  // Subhalo Center-of-Mass in Z
  vector<double> cmzsh(idsh.size(), 0.0);
  for (size_t k = 0; k < idsh.size(); k++) {
    const auto& indices = sh_to_indices[idsh[k]];
    double mzs = 0.0;
    double mtot = 0.0;
    for (int l : indices) {
      mzs += (orgZ[l] * ms4[l]);
      mtot += ms4[l];
    }
    if (mtot != 0.0) {
      cmzsh[k] = 1.0 * mzs / mtot;
    } else {
      cmzsh[k] = 0.0;
    }
  }

  // DEAD LOOP REMOVED: NpTNG loop eliminated per F3!

  // Vectorized population of CMzsh and NpshTNG in O(L)
  vector<double> CMzsh, NpshTNG;
  CMzsh.reserve(idshs.size() + 10);
  NpshTNG.reserve(idshs.size() + 10);
  for (size_t k = 0; k < idsh.size(); k++) {
    const auto& indices = sh_to_indices[idsh[k]];
    double cmz = cmzsh[k];
    double np_val = countSH4[idsh[k]];
    for (size_t idx = 0; idx < indices.size(); idx++) {
      CMzsh.push_back(cmz);
      NpshTNG.push_back(np_val);
    }
  }

  int totPartxy4 = (int)xs.size();
  cout << totPartxy4 << "   type (4) - STAR particles selected in the FoV." << endl;
  if (totPartxy4 > 0) {
    cout << "first subhalo ID: " << idshs[0] << "  |  last subhalo ID: " << idshs[totPartxy4 - 1] << endl;
  }
  cout << endl;

  // 9. Output Generation (Buffered I/O, Deterministic Bit-Exact Preservation)
  if (rdir.empty() || rdir.back() != '/') {
    rdir += "/";
  }
  string coord_path_ = rdir + "coords.lc." + snappl + "_" + std::to_string(target_plane) + ".txt";
  cout << " ... Writing the output file: " << coord_path_ << " ... " << endl;

  ofstream myfile2_;
  vector<char> io_buffer(65536);
  myfile2_.rdbuf()->pubsetbuf(io_buffer.data(), io_buffer.size());
  myfile2_.open(coord_path_);
  if (!myfile2_.is_open()) {
    cerr << "Error: Cannot open output file " << coord_path_ << endl;
    exit(2);
  }

  for (int i = 0; i < totPartxy4; i++) {
    if (idshs[i] > -1) {
      double cmz_out = (i < (int)CMzsh.size()) ? CMzsh[i] : 0.0;
      double np_out  = (i < (int)NpshTNG.size()) ? NpshTNG[i] : 0.0;
      myfile2_ << idshs[i] << " " << xs[i] << " " << ys[i] << " " << zreds[i] << " " 
               << ms4[i] << " " << ims4[i] << " " << mets4[i] << " " << fiub[i] << " " 
               << ages4[i] << " " << zstar[i] << " " << cmz_out << " " << np_out << "\n";
    }
  }
  myfile2_.close();

  cout << ".. Coord file written .." << endl;
  cout << endl;

  double final_rss = getPeakRSS_MB();
  cout << "Peak RSS: " << final_rss << " MB (Enforced Ceiling: " << memory_ceiling_gb << " GB)" << endl;

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  cout << "Execution time in seconds: " << duration.count() * 1e-6 << " s (" << 2.77778e-10 * duration.count() << " h)" << endl;
  cout << "----------------------------------------------------------------------" << endl;

  return 0;
}
