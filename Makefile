ODIR		=	./bin
SFL			=	src/cpp/benchmark_conv.cpp
CPP		 	=	g++
CPP_FLAGS	= 	-g
INC_FLAGS	=	-I./src/include -I./zi
LIB_FLAGS	=
OPT_FLAGS	=	-DNDEBUG -O -std=c++1y
OTH_FLAGS	=	-Wall -Wextra -Wno-unused-result -Wno-unused-local-typedefs
LIBS		=	-lfftw3 -lfftw3f -lpthread -pthread

test: $(SFL)
	$(CPP) -o $(ODIR)/test $(SFL) $(CPP_FLAGS) $(INC_FLAGS) $(LIB_FLAGS) $(OPT_FLAGS) $(OTH_FLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*
