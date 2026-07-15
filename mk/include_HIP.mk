CC   = hipcc
LD = $(CC)

ifeq ($(ENABLE_OPENMP),true)
OPENMP   = -Xcompiler "-fopenmp"
endif

ENABLE_HIP = true
ENABLE_HIP_TEX ?= true

VERSION  = --version
CFLAGS   = -O3 $(OPENMP) -g
HIPCCFLAGS= -std=c++17
LFLAGS   = $(OPENMP)
DEFINES  = -DENABLE_HIP -D_GNU_SOURCE
ifeq ($(ENABLE_HIP_TEX),true)
DEFINES  += -DENABLE_HIP_TEX
endif
INCLUDES =
LIBS     = -lm
