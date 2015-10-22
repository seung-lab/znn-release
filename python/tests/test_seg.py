#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""

# read affinity
fname = "/usr/people/jingpeng/seungmount/research/Jingpeng/09_pypipeline/znn_merged.h5"
import emirt
affs = emirt.emio.imread( fname )

seg = emirt.volume_util.seg_affs(affs, threshold=0.8)

#%%
com = emirt.show.CompareVol((affs[0, :,:,:], seg))
com.vol_compare_slice()

#%%
emirt.show.random_color_show( seg[4,:,:] )