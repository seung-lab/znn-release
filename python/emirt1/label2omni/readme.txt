these files are used to transform raw image and labeling to omni project

usage:

1. python label2omni.py raw_channel_file raw_label_file
2. modify the path of output omni project in "omni.cmd" file. (after the "create:" in first line)
3. run omnification command: bash omnifycation.sh
4. you may need to modify the voxel size in omni.
