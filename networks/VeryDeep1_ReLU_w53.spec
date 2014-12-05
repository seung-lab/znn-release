[INPUT]
size=1

[INPUT_C1]
init_type=Uniform
init_params=0.05
size=3,3,1

[C1]
size=24
activation=relu

[C1_C2]
init_type=Uniform
init_params=0.05
size=3,3,1

[C2]
size=24
activation=relu

[C2_C3]
init_type=Uniform
init_params=0.05
size=2,2,1

[C3]
size=24
activation=relu
filter=max
filter_size=2,2,1
filter_stride=2,2,1

[C3_C4]
init_type=Uniform
init_params=0.05
size=3,3,1

[C4]
size=36
activation=relu

[C4_C5]
init_type=Uniform
init_params=0.05
size=3,3,1

[C5]
size=36
activation=relu
filter=max
filter_size=2,2,1
filter_stride=2,2,1

[C5_C6]
init_type=Uniform
init_params=0.05
size=3,3,1

[C6]
size=48
activation=relu

[C6_C7]
init_type=Uniform
init_params=0.05
size=3,3,1

[C7]
size=48
activation=relu
filter=max
filter_size=2,2,1
filter_stride=2,2,1

[C7_FC]
init_type=Uniform
init_params=0.05
size=3,3,1

[FC]
size=100
activation=relu

[FC_OUTPUT]
init_type=Uniform
init_params=0.05
size=1,1,1

[OUTPUT]
size=2
activation=linear