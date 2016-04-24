# samples example
# the [image] sections indicate the network inputs
# format should be gray images with any bit depth.
#
# [image1]
# fnames =  path/of/image1.tif/h5,
#           path/of/image2.tif/h5
# pp_types = standard2D, none
# is_auto_crop = yes
#
# the [label] sections indicate ground truth of network outputs
# format could be 24bit RGB or gray image with any bit depth.
# the mask images should be binary image with any bit depth.
# only the voxels with gray value greater than 0 is effective for training.
#
# [label1]
# fnames = path/of/image3.tif/h5,
#          path/of/image4.tif/h5
# preprocessing type: one_class, binary_class, none, affinity
# pp_types = binary_class, binary_class
# fmasks = path/of/mask1.tif/h5,
#      path/of/mask2.tif/h5
#
# [sample] section indicates the group of the corresponding input and output labels
# the name should be the same with the one in the network config file
#
# [sample1]
# input1 = 1
# input2 = 2
# output1 = 1
# output2 = 2

[image1]
fnames = ../dataset/test/stack1-image.tif
pp_types = standard2D
is_auto_crop = yes

[image2]
fnames = ../dataset/test/stack2-image.tif
pp_types = standard2D
is_auto_crop = yes

[image3]
fnames = ../dataset/test/stack3-image.tif
pp_types = standard2D
is_auto_crop = yes

[image4]
fnames = ../dataset/test/stack4-image.tif
pp_types = standard2D
is_auto_crop = yes

[label1]
fnames = ../dataset/test/stack1-label.tif
pp_types = binary_class
is_auto_crop = yes
fmasks =

[label2]
fnames = ../dataset/test/stack2-label.tif
pp_types = binary_class
is_auto_crop = yes
fmasks =

[label3]
fnames = ../dataset/test/stack3-label.tif
pp_types = binary_class
is_auto_crop = yes
fmasks =

[label4]
fnames = ../dataset/test/stack4-label.tif
pp_types = binary_class
is_auto_crop = yes
fmasks =

[sample1]
input = 1
output = 1

[sample2]
input = 2
output = 2

[sample3]
input = 3
output = 3

[sample4]
input = 4
output = 4
