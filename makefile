# To build a gibbs sampler for the phone-learning project

CC = icpc
#CFLAGS = -c -Wall -O3 -fomit-frame-pointer -msse2 -mssse3 -fopenmp -parallel 
CFLAGS = -c -xhost -parallel -O3 -fopenmp 
#CC = g++
#CFLAGS = -c -Wall -g
SOURCES = main.cc manager.cc sampler.cc model.cc counter.cc cluster.cc gmm.cc mixture.cc datum.cc segment.cc bound.cc config.cc uniformer.cc toolkit.cc randomizer.cc
OBJECTS=$(SOURCES:.cc=.o)
EXECUTABLE = gibbs

ifeq ($(INTEL_TARGET_ARCH), ia32)
MKL_LINKS=-Wl,--start-group -lmkl_intel -lmkl_intel_thread -lmkl_core -Wl,--end-group -liomp5 -lpthread
else
MKL_LINKS=-Wl,--start-group -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -Wl,--end-group -liomp5 -lpthread
endif

MKL_FLAGS=-I$(MKLROOT)/include -L$(MKLROOT)/lib/$(INTEL_ARCH) $(MKL_LINKS)
IPP_PATHS=-I$(IPPROOT)/include -L$(IPPROOT)/lib/$(INTEL_ARCH)

all: $(SOURCES) $(EXECUTABLE) 

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(MKL_LINKS) 

.cc.o:
	$(CC) $(CFLAGS)  $< -o $@ $(MKL_LINKS) 

clean:
	rm -rf *.o $(EXECUTABLE) 
