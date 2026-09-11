# executable name 
PROG = ./executable_dc

MAIN = main_dc.cpp readTNGParticle.cpp SED.cpp functions.cpp

# Conda paths (dynamically grabbed from your active environment)
CONDA_INC = $(CONDA_PREFIX)/include
CONDA_LIB = $(CONDA_PREFIX)/lib

# Core Compiler and Standard Settings
# Use Conda's environment variable compiler ($CXX) if available, fallback to g++
CXX ?= g++
CXXFLAGS = -std=c++17 -O2 -Wall -I$(CONDA_INC) -I$(CONDA_INC)/eigen3 -I./

# Linker Flags (Tells the loader to find libraries inside Conda at runtime)
LDFLAGS += -L$(CONDA_LIB) -Wl,-rpath,$(CONDA_LIB)

# Explicit dependencies from Conda
LIBS = -larmadillo -lopenblas -llapack \
       -lgsl -lgslcblas \
       -lcfitsio -lCCfits \
       -lfftw3 -lm \
       -lhdf5 -lhdf5_cpp

RM = rm -f -r

default: main

main:
	$(CXX) $(CXXFLAGS) $(MAIN) $(LDFLAGS) $(LIBS) -o $(PROG)

clean:
	$(RM) $(PROG) *.o *~

