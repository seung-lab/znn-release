[INPUT1]
path=./dataset/Ashwin/data/batch3
ext=image
size=512,512,169
pptype=standard2D

[LABEL1]
path=./dataset/Ashwin/data/batch3
ext=label
size=512,512,169
offset=1,1,1
pptype=affinity
ppargs=0.1,0.9

[MASK1]
size=511,511,168
offset=1,1,1
pptype=one
ppargs=3
