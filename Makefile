ODIR		=	./bin
SFL			=	src/cpp/malis_test.cpp
CPP		 	=	g++
ICC			= 	/opt/intel/bin/icc
CPP_FLAGS	= 	-g
INC_FLAGS	=	-I./src/include -I./zi -I.
LIB_FLAGS	=
MKL_FLAGS	=	-static-intel -mkl=sequential -DZNN_USE_MKL_FFT -DZNN_USE_MKL_NATIVE_FFT
OPT_FLAGS	=	-DNDEBUG -O3 -std=c++1y -DZNN_CUBE_POOL_LOCKFREE -DZNN_USE_FLOATS
OTH_FLAGS	=
LIBS		=	-lfftw3 -lfftw3f -lpthread -pthread

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    LIBS += -lrt
endif

test: $(SFL)
	$(CPP) -o $(ODIR)/test $(SFL) $(CPP_FLAGS) $(INC_FLAGS) $(LIB_FLAGS) $(OPT_FLAGS) $(OTH_FLAGS) $(LIBS)

malis:
	$(CPP) -o $(ODIR)/malis src/cpp/malis_test.cpp $(CPP_FLAGS) $(INC_FLAGS) $(LIB_FLAGS) $(OPT_FLAGS) $(OTH_FLAGS) $(LIBS) -Wno-unused-result

malis_debug:
	$(CPP) -o $(ODIR)/malis_debug src/cpp/malis_test.cpp $(CPP_FLAGS) $(INC_FLAGS) $(LIB_FLAGS) $(OPT_FLAGS) $(OTH_FLAGS) $(LIBS) -Wno-unused-result -DDEBUG

.PHONY: clean

mkl: $(SFL)
	$(ICC) -o $(ODIR)/test-mkl $(SFL) $(CPP_FLAGS) $(INC_FLAGS) $(LIB_FLAGS) $(MKL_FLAGS) $(OPT_FLAGS) $(OTH_FLAGS) $(LIBS)

malis: $(SFL)
	$(CPP) -o $(ODIR)/malis $(SFL) $(CPP_FLAGS) $(INC_FLAGS) $(LIB_FLAGS) $(OPT_FLAGS) $(OTH_FLAGS) $(LIBS)
clean:
	rm -f $(ODIR)/*
