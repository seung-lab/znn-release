ODIR		=	.
SFL		=	pyznn.cpp
CPP	 	=	g++
CPP_FLAGS	= 	-g
INC_FLAGS	=	-I../src/include -I../zi -I. -I.. -I/libs/boost/include -I/usr/include/python2.7 -I./Boost.NumPy -I/usr/local/include -I/mnt/bucket/labs/seung/libs/anaconda/include/python2.7
LIB_FLAGS	=	-L/usr/local/lib -L./Boost.NumPy/bin/lib -L/mnt/bucket/labs/seung/libs/anaconda/lib/ -L/usr/local/boost/1.55.0/lib64/
MKL_FLAGS	=	-static-intel -mkl=sequential
OPT_FLAGS	=	-O3 -std=c++1y -DZNN_CUBE_POOL_LOCKFREE -DNDEBUG #-DZNN_USE_FLOATS
OTH_FLAGS	=	-Wall -shared -fPIC -Wno-unused-result -Wno-unused-local-typedefs -DBOOST_DISABLE_ASSERTS
LIBS		=	-lfftw3 -lfftw3f -lpthread -pthread -lpython2.7 -lboost_python -lboost_numpy -ljemalloc

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    LIBS += -lrt
endif

pyznn: $(SFL)
	$(CPP) -o $(ODIR)/pyznn.so $(SFL) $(CPP_FLAGS) $(INC_FLAGS) $(LIB_FLAGS) $(OPT_FLAGS) $(OTH_FLAGS) $(LIBS)

.PHONY: mkl clean

mkl: $(SFL)
	icc -o $(ODIR)/pyznn.so $(SFL) $(MKL_FLAGS) $(INC_FLAGS) $(LIB_FLAGS) $(OPT_FLAGS) $(OTH_FLAGS) $(LIBS)

clean:
	rm -f $(ODIR)/*
