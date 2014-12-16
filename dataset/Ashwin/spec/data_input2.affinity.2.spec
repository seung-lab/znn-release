[INPUT1]
path=./dataset/Ashwin/data/batch2
ext=image
size=512,512,170
pptype=standard2D

[INPUT2]
path=./experiments/Ashwin/2D_boundary/train23_test1/VeryDeep2_w109/unbalanced/eta02_out150/iter_30K/output/out2.1
size=512,512,170
pptype=transform
ppargs=-1,1

[LABEL1]
path=./dataset/Ashwin/data/batch2
ext=label
size=512,512,170
offset=1,1,1
pptype=affinity
ppargs=0.1,0.9

[MASK1]
size=512,512,170
offset=1,1,1
pptype=one
ppargs=3
