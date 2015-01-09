[INPUT1]
path=./dataset/Ashwin/data/batch1
ext=image
size=255,255,168
pptype=standard2D

[INPUT2]
path=./experiments/Ashwin/3D_affinity/train23_test1/VeryDeep2H_w65x9/unbalanced/eta02_out100/iter_30K/output/out1.0
size=255,255,168
pptype=transform
ppargs=-1,1

[INPUT3]
path=./experiments/Ashwin/3D_affinity/train23_test1/VeryDeep2H_w65x9/unbalanced/eta02_out100/iter_30K/output/out1.1
size=255,255,168
pptype=transform
ppargs=-1,1

[INPUT4]
path=./experiments/Ashwin/3D_affinity/train23_test1/VeryDeep2H_w65x9/unbalanced/eta02_out100/iter_30K/output/out1.2
size=255,255,168
pptype=transform
ppargs=-1,1

[LABEL1]
path=./dataset/Ashwin/data/batch1
ext=label
size=255,255,168
offset=1,1,1
pptype=affinity
ppargs=0.1,0.9

[MASK1]
size=255,255,168
offset=1,1,1
pptype=one
ppargs=3
