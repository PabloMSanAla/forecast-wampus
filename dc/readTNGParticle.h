/*
 * @file src/ReadTNGPartcle.h
 * @date 25/09/20
 * @author Erik Romelli - INAF-OATs
 */

#include <vector>
#include <iostream>
#include <H5Cpp.h>
#include <Eigen/Dense>


class readTNGParticle
{

// Define public methods
public:

  // Constructor
  readTNGParticle() = default;

  //Exception Handler
  bool pathExists(hid_t id, const std::string& path);

  // Initializer
  void Initialize(std::string workdir, int particleID);

  // Read the coordinates
  template <class T>  T getHeaderValue(std::string keyword);

  template <class T> std::vector<T> getHeaderValueNonScalar(std::string keyword);

  // Read the coordinates
  std::vector<double> getSnapVectorValue(std::string keyword);

  // // Read the coordinates
  void readCoordinates();

  // Read the Header and get information
  void readHeader(int cutID);

  // Read stars and get information
  void readStars(int cutID);

  // Read subhaloLenType form fof
  void readFof(int cutID);

  // Read info form offset
  void readOffset();

  // Get functions for Header info
  double getBoxSize();
  double getOmegaZero();
  double getOmegaLambda();
  double getTime();
  double getRedshift();
  std::vector<double> getMassTable();
  std::vector<int> getNumPartTotal();

  // Get functions for Stars info
  std::vector<double> getMetallicity();
  std::vector<double> getInitialMass();
  std::vector<double> getStellarFormationTime();
  std::vector<double> getMasses();
  std::vector<double> getX();
  std::vector<double> getY();
  std::vector<double> getZ();

  // Get functions for fof info
  std::vector<int> getStarsLenType();

  // Get functions for fof info
  std::vector<int> getStarsByType();

  // Get functions for fof info for gas
  std::vector<int> getGasLenType();

  // Get functions for fof info for gas
  std::vector<int> getGasByType();

  //GAS//

  // Get gas vector value
  std::vector<double> getGASSnapVectorValue(std::string keyword);

  // Read the GAS coordinates
  void readGASCoordinates();

  // Read the GAS elements abundance
  void readGASElements();

  // Read GAS and get information
  void readGAS(int cutID);

  // Get functions for GAS info
  const std::vector<double>& getGASMetallicity() const { return gmfMetallicity_gas; }
  const std::vector<double>& getGASHIAbundance() const { return HIAbundance_gas; }
  const std::vector<double>& getGASMasses() const { return masses_gas; }
  const std::vector<double>& getGASDensity() const { return density_gas; }
  const std::vector<double>& getGASX() const { return x_gas; }
  const std::vector<double>& getGASY() const { return y_gas; }
  const std::vector<double>& getGASZ() const { return z_gas; }
  const std::vector<double>& getGASH() const { return H_gas; }
  const std::vector<double>& getGASHe() const { return He_gas; }
  const std::vector<double>& getGASC() const { return C_gas; }
  const std::vector<double>& getGASIntEnergy() const { return InternalEnergy_gas; }
  const std::vector<double>& getGASeAbundance() const { return eAbundance_gas; }
  const std::vector<double>& getGASSFR() const { return SFR_gas; }

  //new
  int getDimGas();
  int getDimStars();
  void openStars(int);
  void closeStars();
  void readStars(int, int );
  void readCoordinates(int, int);
  std::vector<double> getSnapVectorValue(std::string, int, int);

  
  // Destructor
  ~readTNGParticle(){
  
    gmfMetallicity.clear();
    gmfInitialMass.clear();
    gmfStellarFormationTime.clear();
    masses.clear();

    gmfMetallicity_gas.clear();
    masses_gas.clear();
    density_gas.clear();
    InternalEnergy_gas.clear();
    eAbundance_gas.clear();
    SFR_gas.clear();
 
  }//= default;

private:

  //snap file name snap_name_template
  std::string snapNameTemplate;

  //snap file name snap_name_template
  std::string fofNameTemplate;

  //snap file name snap_name_template
  std::string offsetFileName;

  // snap object
  H5::H5File snap;

  // fof object
  H5::H5File fof;

  // offset object
  H5::H5File offset;

  // snap header
  H5::Group snapHeader;

  // snap stars
  H5::Group snapStars;

  // boxSize
  double boxSize;

  // omegaZero
  double omegaZero;

  // omegaLambda
  double omegaLambda;

  // time
  double time;

  // redshift
  double redshift;

  // MassTable
  std::vector<double> massTable;

  // numPartTotal
  std::vector<int> numPartTotal;

  // numPartTotal
  std::vector<int> numPartTotal_HW;

  // GFM_Metallicity
  std::vector<double> gmfMetallicity;

  // GFM_InitialMass
  std::vector<double> gmfInitialMass;

  // GFM_stellarFormationTime
  std::vector<double> gmfStellarFormationTime;

  // GFM_stellarFormationTime
  std::vector<double> masses;

  // Coordinates -> X
  std::vector<double> x;

  // Coordinates -> Y
  std::vector<double> y;

  // Coordinates -> Z
  std::vector<double> z;

  // SubhaloLenType -> Stars (4)
  std::vector<int> starsLenType;

  // SnapByType -> Stars (4)
  std::vector<int> starsByType;
  
  // SubhaloLenType -> Gas (0)
  std::vector<int> gasLenType;

  // SnapByType -> Gas (0)
  std::vector<int> gasByType;

  // Coordinates
  Eigen::MatrixXd coordinates;

  // SubhaloLenType
  Eigen::MatrixXd subhaloLenType;

  // SnapByType
  Eigen::MatrixXd snapByType;

  
  //GAS//

  // snap gas
  H5::Group snapGAS;

  // GAS Coordinates
  Eigen::MatrixXd coordinates_gas;

  // GAS Coordinates -> X
  std::vector<double> x_gas;

  // GAS Coordinates -> Y
  std::vector<double> y_gas;

  // GAS Coordinates -> Z
  std::vector<double> z_gas;

  // GAS elements
  Eigen::MatrixXd elements_gas;

  // GAS element abundance -> H
  std::vector<double> H_gas;

  // GAS element abundance -> He
  std::vector<double> He_gas;

  // GAS element abundance -> C
  std::vector<double> C_gas;

  // GAS GFM_Metallicity [no solar met]
  std::vector<double> gmfMetallicity_gas;

  // GAS HIAbundance_gas
  std::vector<double> HIAbundance_gas;

  // GAS mass [1e10 Msun/h]
  std::vector<double> masses_gas;

  // GAS mass [(1e10 Msun/h)/(ckpc/h)^3]
  std::vector<double> density_gas;

  // GAS Internal Energy u [(km/s)^2] - see IllustrisTNG documentation to convert it in Temperature
  std::vector<double> InternalEnergy_gas;

  // GAS ElectronAbundance (n_e= ElectronAbundance*n_H)
  std::vector<double> eAbundance_gas;

  //GAS SFR
  std::vector<double> SFR_gas;

  int dimensionStar;
  int dimensionGas;

}; // readTNGParticle END
