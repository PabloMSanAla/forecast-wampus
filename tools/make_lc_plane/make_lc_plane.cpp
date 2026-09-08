#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstdlib>

using namespace std;

const double speedcunit = 2.99792458e+3;

template <class T>
int locate(const std::vector<T>& v, const T x) {
  size_t n = v.size();
  int jl = -1;
  int ju = n;
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
    return n - 2;
  else
    return jl;
}

double getY(const std::vector<double>& x, const std::vector<double>& y, double xi) {
  int nn = x.size();
  if (x[0] < x[nn - 1]) {
    if (xi > x[nn - 1]) return y[nn - 1];
    if (xi < x[0]) return y[0];
  } else {
    if (xi < x[nn - 1]) return y[nn - 1];
    if (xi > x[0]) return y[0];
  }
  int i = locate(x, xi);
  i = std::min(std::max(i, 0), int(nn) - 2);
  double f = (xi - x[i]) / (x[i + 1] - x[i]);
  if (i > 1 && i < nn - 2) {
    double a0, a1, a2, a3, f2;
    f2 = f * f;
    a0 = y[i + 2] - y[i + 1] - y[i - 1] + y[i];
    a1 = y[i - 1] - y[i] - a0;
    a2 = y[i + 1] - y[i - 1];
    a3 = y[i];
    return a0 * f * f2 + a1 * f2 + a2 * f + a3;
  } else {
    return f * y[i + 1] + (1 - f) * y[i];
  }
}

void printHelp(const char* prog) {
  cout << "Usage: " << prog << " [options]" << endl;
  cout << "Options:" << endl;
  cout << "  -ini <path>     Path to lc.ini (default: looks in FORECAST_LC_INI, ./lc.ini, ../../lc/lc.ini)" << endl;
  cout << "  -o <path>       Output path for planes list (default: from lc.ini PLANES_LIST_FILE, or ./planes_list.txt)" << endl;
  cout << "  -h, --help      Display this help message and exit" << endl;
}

int main(int argc, char** argv) {
  string ini_path = "";
  string output_file = "";

  for (int i = 1; i < argc; i++) {
    string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      printHelp(argv[0]);
      return 0;
    } else if (arg == "-ini" && i + 1 < argc) {
      ini_path = argv[++i];
    } else if (arg == "-o" && i + 1 < argc) {
      output_file = argv[++i];
    } else {
      cerr << "Unknown option: " << arg << endl;
      printHelp(argv[0]);
      return 1;
    }
  }

  // Resolve lc.ini
  if (ini_path.empty()) {
    const char* env_ini = std::getenv("FORECAST_LC_INI");
    if (env_ini != nullptr && env_ini[0] != '\0') {
      ini_path = env_ini;
    } else if (ifstream("lc.ini").good()) {
      ini_path = "lc.ini";
    } else if (ifstream("lc/lc.ini").good()) {
      ini_path = "lc/lc.ini";
    } else if (ifstream("../../lc/lc.ini").good()) {
      ini_path = "../../lc/lc.ini";
    } else {
      ini_path = "lc.ini";
    }
  }

  ifstream inputf(ini_path.c_str());
  if (!inputf.is_open()) {
    cerr << "Error: Cannot open configuration file '" << ini_path << "'." << endl;
    return 1;
  }

  double boxl = 0.0, zs = 0.0, fov = 0.0, res = 0.0;
  string filredshiftlist, filsnaplist, filtimelist, idc, pathsnap, rdir;
  long seedcenter = 0, seedface = 0, seedsign = 0;
  string sim, planes_file_ini;

  string tag;
  inputf >> tag >> boxl;
  inputf >> tag >> zs;
  inputf >> tag >> fov;
  inputf >> tag >> res;
  inputf >> tag >> filredshiftlist;
  inputf >> tag >> filsnaplist;
  inputf >> tag >> filtimelist;
  inputf >> tag >> idc;
  inputf >> tag >> pathsnap;
  inputf >> tag >> rdir;
  inputf >> tag >> seedcenter;
  inputf >> tag >> seedface;
  inputf >> tag >> seedsign;
  inputf >> tag >> sim;
  if (inputf >> tag >> planes_file_ini) {
    // found PLANES_LIST_FILE
  }
  inputf.close();

  if (output_file.empty()) {
    if (!planes_file_ini.empty()) {
      output_file = planes_file_ini;
    } else {
      output_file = "planes_list.txt";
    }
  }

  cout << "============================================================" << endl;
  cout << " FORECAST - Lightcone Plane List Generator (make_lc_plane) " << endl;
  cout << "============================================================" << endl;
  cout << "Config file:      " << ini_path << endl;
  cout << "Box size:         " << boxl << " Mpc/h" << endl;
  cout << "Source redshift:  " << zs << endl;
  cout << "Redshift list:    " << filredshiftlist << endl;
  cout << "Snapshots list:   " << filsnaplist << endl;
  cout << "Comoving dist:    " << idc << endl;
  cout << "Output target:    " << output_file << endl;
  cout << endl;

  // 1. Read redshift list
  ifstream redlist(filredshiftlist.c_str());
  if (!redlist.is_open()) {
    cerr << "Error: redshift list file '" << filredshiftlist << "' not found." << endl;
    return 1;
  }
  vector<double> dtsnaplist, tredlist;
  int buta;
  double butb, butc;
  while (redlist >> buta >> butb >> butc) {
    dtsnaplist.push_back(buta);
    tredlist.push_back(butc);
  }
  redlist.close();

  // 2. Read snapshot list
  ifstream snaplist(filsnaplist.c_str());
  if (!snaplist.is_open()) {
    cerr << "Error: snapshot list file '" << filsnaplist << "' not found." << endl;
    return 1;
  }
  vector<int> lsnap;
  vector<double> lred;
  int s;
  while (snaplist >> s) {
    double zn = getY(dtsnaplist, tredlist, double(s));
    if (zn <= zs) {
      lsnap.push_back(s);
      lred.push_back(zn);
    }
  }
  snaplist.close();

  // 3. Read comoving distance table
  ifstream infiledc(idc.c_str());
  if (!infiledc.is_open()) {
    cerr << "Error: comoving distance file '" << idc << "' not found." << endl;
    return 1;
  }
  vector<double> zl, dl;
  double zi, dci, dli;
  while (infiledc >> zi >> dci >> dli) {
    zl.push_back(zi);
    dl.push_back(dci * speedcunit);
  }
  infiledc.close();

  if (zs > zl.back()) {
    cerr << "Error: source redshift " << zs << " exceeds maximum redshift in comoving distance table (" << zl.back() << ")." << endl;
    return 1;
  }

  double Ds = getY(zl, dl, zs);

  // Field of view validation against simulation box size
  double truefov = fov;
  cout << " " << endl;
  cout << " set the field of view to be square in degrees. " << endl;
  cout << " fov's side  is " << fov << " in degrees and " << fov * 3600.0 << " in arcsec. " << endl;
  double fovradiants = fov / 180.0 * M_PI;
  double truefovradiants = fovradiants;

  // check if the field of view is too large with respect to the box size
  cout << " [ maximum fov side's value allowed " << boxl / Ds * 180.0 / M_PI << " in degrees] " << endl;
  if (fovradiants * Ds > boxl) {
    cout << " field view too large ... I will STOP here!!! " << endl;
    cout << " value set is = " << fov << endl;
    cout << " maximum value allowed " << boxl / Ds * 180.0 / M_PI << " in degrees " << endl;
    cout << " Check it out ... I will STOP here!!! " << endl;
    exit(1);
  }
  cout << endl;

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
  int pl = 0;
  ofstream planelist(output_file.c_str());
  if (!planelist.is_open()) {
    cerr << "Error: Cannot open output file '" << output_file << "' for writing." << endl;
    return 1;
  }

  for (int i = 0; i < nsnaps; i++) {
    breplication.push_back(i);
    pl++;
    planelist << pl << "   " << bzsimlens[i] << "   " << blD[i] << "   " << blD2[i] << "   " << breplication[i] << "   " << bfromsnap[i] << "   " << blred[i] << "\n";
  }

  if (blD2[nsnaps - 1] < Ds - 1e-6) {
    ifstream snaplist_extra(filsnaplist.c_str());
    double s_val;
    while (snaplist_extra >> s_val) {
      if (s_val < bfromsnap[nsnaps - 1]) {
        bfromsnap.push_back(s_val);
        double dlbut = (Ds + blD2[nsnaps - 1]) * 0.5;
        double zbut = getY(dl, zl, dlbut);
        blD.push_back(blD2[nsnaps - 1]);
        blD2.push_back(Ds);
        bzsimlens.push_back(zbut);
        breplication.push_back(breplication[nsnaps - 1] + 1);
        double zn = getY(dtsnaplist, tredlist, double(s_val));
        blsnap.push_back(s_val);
        lred.push_back(zn);
        nsnaps++;
        snaplist_extra.close();
        break;
      }
    }
    pl++;
    planelist << pl << "   " << bzsimlens[nsnaps - 1] << "   " << blD[nsnaps - 1] << "   " << blD2[nsnaps - 1] << "   " << breplication[nsnaps - 1] << "   " << bfromsnap[nsnaps - 1] << "   " << blred[nsnaps - 1] << "\n";
  }

  planelist.close();
  cout << "Successfully generated " << pl << " planes in: " << output_file << endl;
  return 0;
}
