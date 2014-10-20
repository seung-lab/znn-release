[INPUT1]
path=./dataset/ISBI2012/data/batch2
ext=image
size=256,256,30
pptype=standard2D

[LABEL1]
path=./dataset/ISBI2012/data/batch2
ext=label
size=256,256,30
pptype=binary_class

[MASK1]
size=256,256,30
pptype=one
ppargs=2
