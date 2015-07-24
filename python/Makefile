ODIR		=	.
SFL			=	pyznn.cpp
CPP		 	=	g++
CPP_FLAGS	= 	-g 
INC_FLAGS	=	-I../src/include -I../zi -I.. -I/libs/boost/include -I/usr/include/python2.7
LIB_FLAGS	=
OPT_FLAGS	=	-DNDEBUG -O3 -std=c++1y -DZNN_USE_FLOATS
OTH_FLAGS	=	-Wall -shared -fPIC -Wno-unused-result -Wno-unused-local-typedefs
LIBS		=	-lboost_python -lfftw3 -lfftw3f -lpthread -pthread

pyznn: $(SFL)
	$(CPP) -o $(ODIR)/pyznn.so $(SFL) $(CPP_FLAGS) $(INC_FLAGS) $(LIB_FLAGS) $(OPT_FLAGS) $(OTH_FLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*
