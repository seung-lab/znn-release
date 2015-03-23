ODIR		=	./bin
CPP		=	g++
CPP_FLAGS	= 	-g
INC_FLAGS	=	-I. -I./src -I./zi
OPT_FLAGS	=	-DNDEBUG -O3 -std=c++11
OTH_FLAGS	=	-Wall -Wextra -Wno-unused-result -Wno-unused-local-typedefs

LIBS		=	-lfftw3 -lpthread -lrt -lfftw3_threads
BOOST_LIBS	=	-lboost_program_options -lboost_regex -lboost_filesystem -lboost_system

znn: src/main.cpp
	$(CPP) -o $(ODIR)/znn src/main.cpp $(CPP_FLAGS) $(INC_FLAGS) $(OPT_FLAGS) $(OTH_FLAGS) $(LIBS) $(BOOST_LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*
