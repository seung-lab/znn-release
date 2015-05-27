g++ -std=c++1y _test2d_training.cpp -I../../.. -ljemalloc -lfftw3 -lpthread -O3 -DNDEBUG -o bin/test2d_training
g++ -std=c++1y _test2d_forward.cpp -I../../.. -ljemalloc -lfftw3 -lpthread -O3 -DNDEBUG -o bin/test2d_forward
g++ -std=c++1y _test2d_training_strided.cpp -I../../.. -ljemalloc -lfftw3 -lpthread -O3 -DNDEBUG -o bin/test2d_training_strided
g++ -std=c++1y _test3d_training.cpp -I../../.. -ljemalloc -lfftw3 -lpthread -O3 -DNDEBUG -o bin/test3d_training
g++ -std=c++1y _test3d_forward.cpp -I../../.. -ljemalloc -lfftw3 -lpthread -O3 -DNDEBUG -o bin/test3d_forward
g++ -std=c++1y _test3d_training_strided.cpp -I../../.. -ljemalloc -lfftw3 -lpthread -O3 -DNDEBUG -o bin/test3d_training_strided
