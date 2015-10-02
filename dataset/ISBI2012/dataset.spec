# samples example
# [image1]
# fnames =  path/of/image1.tif/h5,
#           path/of/image2.tif/h5
# pp_types = standard2D, none
# is_auto_crop = yes
# [label1]
# fnames = path/of/image3.tif/h5,
#          path/of/image4.tif/h5
# preprocessing type: one_class, binary_class, none, affinity
# pp_types = binary_class, binary_class
# fmasks = path/of/mask1.tif/h5,
#	   path/of/mask2.tif/h5
#
# [sample1]
# the name should be the same with the one in the network config file
# input1 = 1
# input2 = 2
# output1 = 1
# output2 = 2

[image1]
fnames = ../dataset/ISBI2012/train-volume.tif
pp_types = standard2D
is_auto_crop = yes

[label1]
fnames = ../dataset/ISBI2012/train-labels.tif
pp_types = auto
is_auto_crop = yes
fmasks =

[sample1]
input = 1
output = 1

[image2]
fnames = ../dataset/ISBI2012/test-volume.tif
pp_types = standard2D
is_auto_crop = yes

[sample2]
input = 2
