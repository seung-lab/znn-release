ODIR		=	.
SFL 		=	pymalis.cpp
CPP 		=	g++
CPP_FLAGS	= 	-g
INC_FLAGS	=	-I/opt/boost/include -I../../src/include -I../../zi -I. -I.. -I../../ -I/usr/include/python2.7 -I./Boost.NumPy -I/usr/local/include -I/opt/anaconda/include/python2.7 -I/mnt/bucket/labs/seung/libs/anaconda/include/python2.7
LIB_FLAGS	=	-L/opt/boost/lib -L/usr/local/lib -L./Boost.NumPy/bin/lib -L/mnt/bucket/labs/seung/libs/anaconda/lib/ -L/usr/local/boost/1.55.0/lib64/ -L/opt/anaconda/lib
MKL_FLAGS	=	-static-intel -mkl=sequential -DZNN_USE_MKL_FFT -DZNN_USE_MKL_NATIVE_FFT
OPT_FLAGS	=	-O3 -std=c++1y -DZNN_CUBE_POOL_LOCKFREE -DZNN_USE_FLOATS -DNDEBUG
OTH_FLAGS	=	-Wall -shared -fPIC -Wno-unused-result -Wno-unused-local-typedefs -DBOOST_DISABLE_ASSERTS
LIBS		=	-lfftw3 -lfftw3f -lpthread -pthread -lpython2.7 -lboost_python -lboost_numpy

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    LIBS += -lrt
endif

pymalis: $(SFL)
	$(CPP) -o $(ODIR)/pymalis.so $(SFL) $(CPP_FLAGS) $(INC_FLAGS) $(LIB_FLAGS) $(OPT_FLAGS) $(OTH_FLAGS) $(LIBS)
