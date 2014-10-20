[INPUT]
size=1

[INPUT_C1]
size=4,4,1
init_type=Uniform
init_params=0.05

[C1]
size=48
activation=tanh
act_params=1.7159,0.6666
filter=max
filter_size=2,2,1
filter_stride=2,2,1

[C1_C2]
size=5,5,1
init_type=Uniform
init_params=0.05

[C2]
size=48
activation=tanh
act_params=1.7159,0.6666
filter=max
filter_size=2,2,1
filter_stride=2,2,1

[C2_C3]
size=4,4,1
init_type=Uniform
init_params=0.05

[C3]
size=48
activation=tanh
act_params=1.7159,0.6666
filter=max
filter_size=2,2,1
filter_stride=2,2,1

[C3_C4]
size=4,4,1
init_type=Uniform
init_params=0.05

[C4]
size=48
activation=tanh
act_params=1.7159,0.6666
filter=max
filter_size=2,2,1
filter_stride=2,2,1

[C4_FC]
size=3,3,1
init_type=Uniform
init_params=0.05

[FC]
size=200
activation=tanh
act_params=1.7159,0.6666

[FC_OUTPUT]
init_type=Uniform
init_params=0.05
size=1,1,1

[OUTPUT]
size=2
activation=linear
act_params=1,0