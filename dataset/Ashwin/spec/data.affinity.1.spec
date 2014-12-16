[INPUT1]
path=./dataset/Ashwin/data/batch1
ext=image
size=255,255,168
pptype=standard2D

[LABEL1]
path=./dataset/Ashwin/data/batch1
ext=label
size=255,255,168
offset=1,1,1
pptype=offset
ppargs=0.1,0.9

[MASK1]
size=254,254,167
offset=1,1,1
pptype=one
ppargs=3
