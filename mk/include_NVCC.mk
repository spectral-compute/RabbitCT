CC   = nvcc
LD = $(CC)

ifeq ($(ENABLE_OPENMP),true)
OPENMP   = -Xcompiler "-fopenmp"
endif

ENABLE_CUDA = true
ENABLE_CUDA_TEX ?= true

VERSION  = --version
CFLAGS   = -O3 $(OPENMP) -g
NVCCFLAGS= -std=c++17
LFLAGS   = $(OPENMP)
DEFINES  = -DENABLE_CUDA -D_GNU_SOURCE
ifeq ($(ENABLE_CUDA_TEX),true)
DEFINES  += -DENABLE_CUDA_TEX
endif
INCLUDES =
LIBS     = -lm
