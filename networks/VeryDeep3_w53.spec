[INPUT]
size=1

[INPUT2]
size=1

[INPUT_C1]
init_type=Uniform
init_params=0.05
size=3,3,1

[INPUT2_F1]
init_type=Uniform
init_params=0.05
size=3,3,1

[C1]
size=24
activation=relu

[F1]
size=24
activation=relu

[C1_C2]
init_type=Uniform
init_params=0.05
size=3,3,1

[F1_F2]
init_type=Uniform
init_params=0.05
size=3,3,1

[C2]
size=24
activation=relu

[F2]
size=24
activation=relu

[C2_C3]
init_type=normalized
size=2,2,1

[F2_F3]
init_type=normalized
size=3,3,1

[C3]
size=24
activation=tanh
act_params=1.7159,0.6666
filter=max
filter_size=2,2,1
filter_stride=2,2,1

[F3]
size=24
activation=tanh
act_params=1.7159,0.6666

[C3_C4]
init_type=Uniform
init_params=0.05
size=3,3,1

[F3_F4]
init_type=Uniform
init_params=0.05
size=3,3,1

[C4]
size=36
activation=relu

[F4]
size=24
activation=relu

[C4_C5]
init_type=normalized
size=3,3,1

[F4_F5]
init_type=normalized
size=3,3,1

[C5]
size=36
activation=tanh
act_params=1.7159,0.6666
filter=max
filter_size=2,2,1
filter_stride=2,2,1

[F5]
size=24
activation=tanh
act_params=1.7159,0.6666

[C5_C6]
init_type=Uniform
init_params=0.05
size=3,3,1

[F5_F6]
init_type=Uniform
init_params=0.05
size=3,3,1

[C6]
size=48
activation=relu

[F6]
size=24
activation=relu

[C6_C7]
init_type=normalized
size=3,3,1

[F6_F7]
init_type=normalized
size=3,3,1

[C7]
size=48
activation=tanh
act_params=1.7159,0.6666
filter=max
filter_size=2,2,1
filter_stride=2,2,1

[F7]
size=24
activation=tanh
act_params=1.7159,0.6666

[C7_C8]
init_type=Uniform
init_params=0.05
size=3,3,1

[F7_F8]
init_type=Uniform
init_params=0.05
size=3,3,1

[C8]
size=48
activation=relu

[F8]
size=24
activation=relu

[C8_FC]
init_type=Uniform
init_params=0.05
size=1,1,1

[F8_FC]
init_type=Uniform
init_params=0.05
size=1,1,1

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