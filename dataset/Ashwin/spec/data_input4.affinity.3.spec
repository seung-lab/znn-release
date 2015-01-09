[INPUT1]
path=./dataset/Ashwin/data/batch3
ext=image
size=512,512,169
pptype=standard2D

[INPUT2]
path=./experiments/Ashwin/3D_affinity/train23_test1/VeryDeep2H_w65x9/unbalanced/eta02_out100/iter_30K/output/out3.0
size=512,512,169
pptype=transform
ppargs=-1,1

[INPUT3]
path=./experiments/Ashwin/3D_affinity/train23_test1/VeryDeep2H_w65x9/unbalanced/eta02_out100/iter_30K/output/out3.1
size=512,512,169
pptype=transform
ppargs=-1,1

[INPUT4]
path=./experiments/Ashwin/3D_affinity/train23_test1/VeryDeep2H_w65x9/unbalanced/eta02_out100/iter_30K/output/out3.2
size=512,512,169
pptype=transform
ppargs=-1,1

[LABEL1]
path=./dataset/Ashwin/data/batch3
ext=label
size=512,512,169
offset=1,1,1
pptype=affinity
ppargs=0.1,0.9

[MASK1]
size=512,512,169
offset=1,1,1
pptype=one
ppargs=3
