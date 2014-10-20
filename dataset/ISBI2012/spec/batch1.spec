[INPUT1]
path=./dataset/ISBI2012/data/batch1
ext=image
size=512,512,30
pptype=standard2D

[LABEL1]
path=./dataset/ISBI2012/data/batch1
ext=label
size=512,512,30
pptype=binary_class

[MASK1]
size=512,512,30
pptype=one
ppargs=2
