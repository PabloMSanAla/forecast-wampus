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
#include <limits>
#include <memory>
#include <valarray>
#include <CCfits/CCfits>
#include <CCfits/FitsError.h>
#include "functions.h"


/***************************************************************************/
/*                                                                         */
/*            postprocessing FORECAST - creating the raw image             */
/*                                                                         */
/*  if you use it or do any mods, please cite Fortuni et al. (2023).       */
/*                                         flaminia.fortuni@inaf.it        */
/*                                                                         */
/*                                                                         */  
/*  this is a postprocessing module for the FORECAST code;                 */
/*  it creates the raw image from the particle catalog                     */
/*  it runs on one snapshot/plane per time.                                */
/*  - input: particle catalog created from either df,dc or igm module      */
/*  - output: raw image in the chosen filter, one snapshot/plane per time  */ 
/*                                                                         */
/*                                                                         */
/*  for a comprehensive guide of FORECAST, visit                           */
/*                          https://github.com/flaminiafortuni/FORECAST    */
/*  for a full description of the software                                 */     
/*       https://ui.adsabs.harvard.edu/abs/2023arXiv230519166F/abstract    */
/*                                                                         */   
/***************************************************************************/


using namespace std;
using namespace CCfits;

const double h0=0.6774;


int main(int argc, char** argv){

  cout << "----------------------------------------------------------------------" << endl;
  cout << " " << endl; 
  cout << "   ------------------------------------------------------ " << endl;
  cout << "   -                                                    - " << endl;
  cout << "   -           2D Mapping Simulation Snapshot           - " << endl;
  cout << "   -                                                    - " << endl;
  cout << "   -               creating the final image             - " << endl;
  cout << "   ------------------------------------------------------ " << endl;
  // 1. Check CLI arguments
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
      cout << "Usage: " << argv[0] << " <snapshot> <plane_number> [-ini <image.ini>]" << endl;
      exit(0);
    } else {
      positional_args.push_back(arg);
    }
  }

  if (positional_args.size() < 2) {
    cerr << "Usage: " << argv[0] << " <snapshot> <plane_number> [-ini <image.ini>]" << endl;
    exit(1);
  }

  string snappl = positional_args[0];
  string plane = positional_args[1];
  int sourceID = std::atoi(snappl.c_str());
  int iplrestart = std::atoi(plane.c_str());

  // check if the restart file exists (for legacy compatibility)
  std::string fileplstart = plane + ".d";
  std::ifstream infileplstart;
  infileplstart.open(fileplstart.c_str());
  if (infileplstart.is_open()) {
    cout << " " << endl;
    cout << " I will read the restart file >> " << fileplstart << endl;
    infileplstart >> iplrestart;
    cout << " iplrestart = " << iplrestart << endl;
    cout << " " << endl;
    infileplstart.close();
  }

  // 2. Read Configuration from image.ini
  float fov, res;
  string filter, filfilters, planes_file, module, catpath, rdir;

  readParameters(&fov, &res,
                 &filter,
                 &filfilters,
                 &planes_file,
                 &module,
                 &catpath,
                 &rdir,
                 custom_ini);


  // 3. Resolve and read planes_list.txt from configuration
  int n_pl = 0;
  float blD = 0.0f, blD2 = 0.0f, zsim = 0.0f, invh0 = 1.0f / h0;
  bool plane_matched = false;

  ifstream oplane;  
  oplane.open(planes_file.c_str());
  if (oplane.is_open()) {
    int npl, reppl, blsnappl;
    float zpl, blDpl, blD2pl, zpltrue;
    
    while (oplane >> npl >> zpl >> blDpl >> blD2pl >> reppl >> blsnappl >> zpltrue) {
      if ((reppl == iplrestart || npl - 1 == iplrestart) && blsnappl == sourceID) {
        n_pl = npl;
        blD = blDpl;
        blD2 = blD2pl;
        zsim = zpl;
        plane_matched = true;
        break;
      }
    }
    oplane.close();
  } else {
    cerr << "Error: planes list file " << planes_file << " could not be opened." << endl;
    exit(2);
  }

  if (!plane_matched) {
    cerr << "Error: Snapshot " << sourceID << " and plane " << iplrestart
         << " do not match any entry in " << planes_file << "." << endl;
    exit(1);
  }

  // Pixels of the image
  unsigned long int truenpix = int(fov * 3600.0 / res);
  int bufferpix = int(ceil((truenpix + 1) * 20 / 14142));  // add bufferpix/2 on each side
  unsigned long int npix = truenpix + bufferpix;

  cout << "N. pixels: " << truenpix << "; buffer pixels: " << bufferpix << endl;
  cout << endl;

  vector<float> idshs(0), xs(0), ys(0), fH(0);
  vector<vector<double>> fluxmap;

  cout << "... Reading filter list from " << filfilters << " ..." << endl;

  // 4. Read filter list from configuration
  ifstream listfilter;
  listfilter.open(filfilters.c_str());
  vector<string> nfilter;
  if (listfilter.is_open()) {
    string buta;
    while (listfilter >> buta) {
      nfilter.push_back(buta);
    }
    listfilter.close();
  } else {
    cerr << "Error: filter list file " << filfilters << " could not be opened. Please check FILTERS_FILE in image.ini." << endl;
    exit(2);
  }

  // Find filter position in filter list
  int pos = -1;
  int Nfilters = (int)nfilter.size();
  for (int i = 0; i < Nfilters; i++) {
    if (nfilter[i] == filter) {
      pos = i;
      break;
    }
  }

  if (pos == -1) {
    cerr << "Error: filter '" << filter << "' not found in filter list file " << filfilters << "." << endl;
    exit(1);
  }

  // 5. Read input particle catalog from configured directory
  string filoutcat = catpath + "flux." + module + "." + snappl + "_" + plane + ".txt";

  if (module == "dc" || module == "igm") {
    cout << " " << endl;
    cout << "... Reading input file (" << filoutcat << ") from " << module << " module ..." << endl;
    cout << " .. content: #(ids,x,y,redshift,mass,intial mass,metallicity, age) for stellar particles;" << endl;
    cout << " ..          #(Zgas_w, NHIgas_w) computed on galaxy-basis, for galaxies with id==ids." << endl;
    cout << " ..          #(Np_sh, N_sim): number of particles with ids in the lightcone, number of particle with ids in the original simulation." << endl;
    cout << " ..          #(reddened flux in Nfilters) for stellar particles." << endl;
    cout << endl;
    
    ifstream ocf;
    ocf.open(filoutcat.c_str());  
    if (ocf.is_open()) {
      int shid, npsh, npshtng;
      float xi, yi, zi, zri, mi, imi, ai, Zmi;
      double NHImi;
      long double meti;
      while (ocf >> shid >> xi >> yi >> zri >> mi >> imi >> meti >> ai >> Zmi >> NHImi >> npsh >> npshtng) {
        idshs.push_back(shid);
        xs.push_back(xi);
        ys.push_back(yi);
        std::vector<double> temp_flux;
        double temp_val;
        int cf = 0;
        while (cf < Nfilters) {
          if (ocf >> temp_val) {
            temp_flux.push_back(temp_val);
            cf += 1;
          } else {
            break;
          }
        }
        fluxmap.push_back(temp_flux);
        ocf.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      }    
      ocf.close();
    } else {
      cerr << "Error: catalog file " << filoutcat << " does not exist. Please check PATH_WHERE_CATALOGUES_ARE_LOCATED in image.ini." << endl;
      exit(2);
    }
  } else if (module == "df") {
    cout << "... Reading dust-free particle-based catalog (" << filoutcat << ") produced in the " << module << " module ..." << endl;
    cout << " .. content: #(ids,x,y,redshift,mass,intial mass,metallicity, CM_sh, age, N_sim) for stellar particles;" << endl;
    cout << " ..          #(dust-free flux in Nfilters) for stellar particles." << endl;
    cout << endl;
    
    ifstream ocf;
    ocf.open(filoutcat.c_str());
    if (ocf.is_open()) {
      int shid, NPTNG;
      float xi, yi, zi, zri, mi, imi, ai, fi, sni, cmzsh;
      long double meti, mabi;
      while (ocf >> shid >> xi >> yi >> zri >> mi >> imi >> meti >> cmzsh >> ai >> NPTNG) { 
        idshs.push_back(shid);
        xs.push_back(xi);
        ys.push_back(yi);
        std::vector<double> temp_flux;
        double temp_val;
        int cf = 0;
        while (cf < Nfilters) {
          if (ocf >> temp_val) {
            temp_flux.push_back(temp_val);
            cf += 1;
          } else {
            break;
          }
        }
        fluxmap.push_back(temp_flux);
        ocf.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      }    
      ocf.close();
    } else {
      cerr << "Error: catalog file " << filoutcat << " does not exist. Please check PATH_WHERE_CATALOGUES_ARE_LOCATED in image.ini." << endl;
      exit(2);
    }
  } else {
    cerr << "Error: Unknown module '" << module << "'. Must be df, dc, or igm." << endl;
    exit(1);
  }

  // collecting fluxes for the final image
  for (size_t j = 0; j < fluxmap.size(); j++) {
    fH.push_back(fluxmap[j][pos]);
  }

  std::valarray<float> mapxy4(npix * npix);
  string pixu = "flux [uJ]";
  string fileoutput = rdir + filter + "." + module + "." + snappl + "_" + plane + ".fits";
  
  //make the map
  mapxy4=gridist_nok(xs,ys,fH,npix);
  int ntotxy4= xs.size();
  cout << "N. particles: " << xs.size() << " at snapshot " << snappl << " and plane " << plane << endl;
  cout << endl;
  if(ntotxy4>0){
    
          long naxis = 2;
	  long naxes[2]={ static_cast<long>(truenpix), static_cast<long>(truenpix) };
	  string count="1";
	  
	  std::unique_ptr<FITS> ffxy( new FITS(fileoutput, FLOAT_IMG, naxis, naxes ) ); 
	  std::vector<long> naxex( 2 );
	  naxex[0]=truenpix;
	  naxex[1]=truenpix;
	  PHDU *phxy=&ffxy->pHDU();
	  // phxy->write( 1, truenpix*truenpix, mapxytot4 );
	  valarray<float> pmap(truenpix*truenpix);
	  pmap = rescalemap(mapxy4,npix,truenpix);
	  cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << endl;
	  cout << "@  ... Writing .fits file " << fileoutput << " ... @" << endl;
	  cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << endl;
	  cout << " " << endl;
	  long  fpixel(1);
	  phxy->write( fpixel, truenpix*truenpix , pmap );
	  phxy->addKey ("N_PIXEL_BY_SIDE",truenpix," ");
	  phxy->addKey ("BUFFER_PIXELS",bufferpix," ");
	  phxy->addKey ("PIXELUNIT",pixu," ");
	  phxy->addKey ("N_STELLAR_PARTICLES",ntotxy4," ");
	  phxy->addKey ("FILTER",filter," ");	  
	  phxy->addKey ("REDSHIFT",zsim," "); 
	  phxy->addKey ("PHYSICALSIZE_BY_SIDE",fov,"degrees "); 
	  phxy->addKey ("Dl_LOW",blD*invh0,"comoving distance in Mpc"); 
	  phxy->addKey ("Dl_UP",blD2*invh0,"comoving distance in Mpc");
	  phxy->addKey ("HUBBLE",h0," ");
  }
  cout << "--------------------------" << endl;
}


