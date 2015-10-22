export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/opt/people/zlateski/caffe/build/lib

./caffe/build/tools/caffe time --model=./becnhmarks/2d10_1.proto --iterations=100 --gpu 0
./caffe/build/tools/caffe time --model=./becnhmarks/2d10_2.proto --iterations=100 --gpu 0
./caffe/build/tools/caffe time --model=./becnhmarks/2d10_4.proto --iterations=100 --gpu 0
./caffe/build/tools/caffe time --model=./becnhmarks/2d10_8.proto --iterations=100 --gpu 0
./caffe/build/tools/caffe time --model=./becnhmarks/2d10_16.proto --iterations=100 --gpu 0
./caffe/build/tools/caffe time --model=./becnhmarks/2d10_32.proto --iterations=100 --gpu 0
./caffe/build/tools/caffe time --model=./becnhmarks/2d10_64.proto --iterations=100 --gpu 0

./caffe/build/tools/caffe time --model=./becnhmarks/2d20_1.proto --iterations=100 --gpu 0
./caffe/build/tools/caffe time --model=./becnhmarks/2d20_2.proto --iterations=100 --gpu 0
./caffe/build/tools/caffe time --model=./becnhmarks/2d20_4.proto --iterations=100 --gpu 0
./caffe/build/tools/caffe time --model=./becnhmarks/2d20_8.proto --iterations=100 --gpu 0
./caffe/build/tools/caffe time --model=./becnhmarks/2d20_16.proto --iterations=100 --gpu 0
./caffe/build/tools/caffe time --model=./becnhmarks/2d20_32.proto --iterations=100 --gpu 0
./caffe/build/tools/caffe time --model=./becnhmarks/2d20_64.proto --iterations=100 --gpu 0

./caffe/build/tools/caffe time --model=./becnhmarks/2d30_1.proto --iterations=100 --gpu 0
./caffe/build/tools/caffe time --model=./becnhmarks/2d30_2.proto --iterations=100 --gpu 0
./caffe/build/tools/caffe time --model=./becnhmarks/2d30_4.proto --iterations=100 --gpu 0
./caffe/build/tools/caffe time --model=./becnhmarks/2d30_8.proto --iterations=100 --gpu 0
./caffe/build/tools/caffe time --model=./becnhmarks/2d30_16.proto --iterations=100 --gpu 0
./caffe/build/tools/caffe time --model=./becnhmarks/2d30_32.proto --iterations=100 --gpu 0
./caffe/build/tools/caffe time --model=./becnhmarks/2d30_64.proto --iterations=100 --gpu 0

./caffe/build/tools/caffe time --model=./becnhmarks/2d40_1.proto --iterations=100 --gpu 0
./caffe/build/tools/caffe time --model=./becnhmarks/2d40_2.proto --iterations=100 --gpu 0
./caffe/build/tools/caffe time --model=./becnhmarks/2d40_4.proto --iterations=100 --gpu 0
./caffe/build/tools/caffe time --model=./becnhmarks/2d40_8.proto --iterations=100 --gpu 0
./caffe/build/tools/caffe time --model=./becnhmarks/2d40_16.proto --iterations=100 --gpu 0
./caffe/build/tools/caffe time --model=./becnhmarks/2d40_32.proto --iterations=100 --gpu 0
./caffe/build/tools/caffe time --model=./becnhmarks/2d40_64.proto --iterations=100 --gpu 0
