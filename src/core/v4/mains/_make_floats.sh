g++ -std=c++1y _test2d_training.cpp -I../../../.. -I.. -ljemalloc -lfftw3f -lpthread -O3 -DNDEBUG -DZNN_CUBE_POOL -DZNN_USE_FLOATS -o ../bin/test2d_training
g++ -std=c++1y _test2d_forward.cpp -I../../../.. -I.. -ljemalloc -lfftw3f -lpthread -O3 -DNDEBUG -DZNN_CUBE_POOL -DZNN_USE_FLOATS -o ../bin/test2d_forward
g++ -std=c++1y _test2d_training_strided.cpp -I../../../.. -I.. -ljemalloc -lfftw3f -lpthread -O3 -DNDEBUG -DZNN_CUBE_POOL -DZNN_USE_FLOATS -o ../bin/test2d_training_strided
g++ -std=c++1y _test3d_training.cpp -I../../../.. -I.. -ljemalloc -lfftw3f -lpthread -O3 -DNDEBUG -DZNN_CUBE_POOL -DZNN_USE_FLOATS -o ../bin/test3d_training
g++ -std=c++1y _test3d_forward.cpp -I../../../.. -I.. -ljemalloc -lfftw3f -lpthread -O3 -DNDEBUG -DZNN_CUBE_POOL -DZNN_USE_FLOATS -o ../bin/test3d_forward
g++ -std=c++1y _test3d_training_strided.cpp -I../../../.. -I.. -ljemalloc -lfftw3f -lpthread -O3 -DNDEBUG -DZNN_CUBE_POOL -DZNN_USE_FLOATS -o ../bin/test3d_training_strided
