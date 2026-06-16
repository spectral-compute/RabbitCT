CC   = nvcc
LD = $(CC)

ifeq ($(ENABLE_OPENMP),true)
OPENMP   = -Xcompiler "-fopenmp"
endif

ENABLE_CUDA = true

VERSION  = --version
CFLAGS   = -O3 $(OPENMP) -g
NVCCFLAGS= -std=c++17
LFLAGS   = $(OPENMP)
DEFINES  = -DENABLE_CUDA -D_GNU_SOURCE
INCLUDES =
LIBS     = -lm
