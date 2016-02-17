#!/usr/bin/env python
__doc__ = """

Read-only/writable TensorData classes.

Kisuk Lee <kisuklee@mit.edu>, 2015
"""

from vector import Vec3d, minimum, maximum
from box import *

import numpy as np
import math

class TensorData:
    """
    Read-oly tensor data.
    This is not a general tensor, in that 4th dimension is only regarded as
    parallel channels, not actual dimension for arbitrary access.
    Threfore, every data access is made through 3-dimensional vector.

    Methods
    -------
    get_value
    get_patch
    get_volume

    Members
    -------
    _data       -- numpy ndarray
    _dim        -- dimension of volume data (update dependency: _data)
    _FoV        -- field of view (patch size)
    _offset     -- offset from origin
    _bb         -- bounding box (update dependency: _dim, _offset)
    _rg         -- range (update dependency: _dim, _offset, _FoV)

    """

    def __init__(self, data, FoV=(0,0,0), offset=(0,0,0)):
        """
        Initialize a tensor data.
        """
        # initialization
        self._FoV    = Vec3d(0,0,0)
        self._offset = Vec3d(0,0,0)

        # set data (order is important)
        self._set_data(data)
        self._set_FoV(FoV)
        self._set_offset(offset)

    def get_patch(self, pos):
        """
        Extract a subvolume of size _FoV centered on pos.
        """
        assert self._rg.contains(pos)
        pos -= self._offset # local coordinate system
        box  = centered_box(pos, self._FoV)
        vmin = box.min()
        vmax = box.max()
        return np.copy(self._data[:,vmin[0]:vmax[0],
                                    vmin[1]:vmax[1],
                                    vmin[2]:vmax[2]])

    def get_volume(self):
        return self._data

    def size(self):
        """
        Return the shape of tensor data.
        E.g. (w,z,y,x), where w is the number of parallel channels.
        """
        return self._data.shape

    def dimension(self):
        """
        Return the shape of accessible dimensions.
        E.g. (z,y,x), w is not an accessible dimension.
        """
        return self._dim

    def FoV(self):
        return Vec3d(self._FoV)

    def offset(self):
        return Vec3d(self._offset)

    def bounding_box(self):
        return Box(self._bb)

    def range(self):
        return Box(self._rg)

    # private methods for setting members
    def _check_data(self, data):
        assert isinstance(data, np.ndarray)
        assert data.ndim == 4

        # return data dimension
        return Vec3d(data.shape[1:])

    def _set_data(self, data):
        # TODO: should deep copy data?
        self._data = data
        self._dim  = self._check_data(data)

        # update bounding box & range
        self._set_bounding_box()
        self._set_range()

    def _set_FoV(self, FoV):
        """
        Set a nonnegative field of view (FoV), i.e., patch size.
        If FoV is zero, set it to cover the whole volume.
        """
        FoV = Vec3d(FoV)
        if FoV == (0,0,0):
            FoV = Vec3d(self._dim)

        # FoV should be nonnegative and smaller than data volume
        self._FoV = minimum(maximum(FoV,(0,0,0)), self._dim)

        # update range
        self._set_range()

    def _set_offset(self, offset):
        """
        Set an offset from origin.
        """
        self._offset = Vec3d(offset)

        # update bounding box & range
        self._set_bounding_box()
        self._set_range()

    def _set_bounding_box(self):
        """
        Set a bounding box.
        """
        self._bb = Box((0,0,0),self._dim)
        self._bb.translate(self._offset)

    def _set_range(self):
        """
        Set a valid range for extracting patches.
        """
        # top margin
        top = self._FoV/2
        # bottom margin
        btm = self._FoV - top - (1,1,1)

        vmin = self._offset + top
        vmax = self._offset + self._dim - btm

        self._rg = Box(vmin,vmax)

    # String representaion (for printing and debugging)
    def __str__( self ):
        return "<TensorData>\nsize = %s\ndim  = %s\nFoV  = %s\noff  = %s\n" % \
               (self.size(), self._dim, self._FoV, self._offset)


class WritableTensorData(TensorData):
    """
    Writable tensor data.

    """
    def __init__(self, data_or_shape, FoV=(0,0,0), offset=(0,0,0)):
        """
        Initialize a writable tensor data, or create a new tensor of zeros.
        """
        if isinstance(data_or_shape, np.ndarray):
            TensorData.__init__(self,data_or_shape,FoV,offset)
        else:
            shape = Vec3d(data_or_shape)
            data  = np.zeros(shape)
            TensorData.__init__(self,data,FoV,offset)

    def set_patch(self, pos, patch):
        """
        Write a patch of size _FoV centered on pos.
        """
        assert self._rg.contains(pos)
        dim = self._check_data(patch)
        assert dim == self._FoV
        box = centered_box(pos, dim)

        # local coordinate
        box.translate(-self._offset)
        vmin = box.min()
        vmax = box.max()
        self._data[:,vmin[0]:vmax[0],vmin[1]:vmax[1],vmin[2]:vmax[2]] = patch


########################################################################
## Unit Testing                                                       ##
########################################################################
if __name__ == "__main__":

    import unittest

    ####################################################################
    class UnitTestTensorData(unittest.TestCase):

        def setUp(self):
            pass

        def testCreation(self):
            data = np.zeros((4,4,4,4))
            v = TensorData(data, (3,3,3), (1,1,1))
            self.assertTrue(v.size()   == (4,4,4,4))
            self.assertTrue(v.FoV()    == (3,3,3))
            self.assertTrue(v.offset() == (1,1,1))
            bb = v.bounding_box()
            rg = v.range()
            self.assertTrue(bb == Box((1,1,1),(5,5,5)))
            self.assertTrue(rg == Box((2,2,2),(4,4,4)))

    ####################################################################
    class UnitTestWritableTensorData(unittest.TestCase):

        def setUp(self):
            pass

        def testCreation(self):
            data = np.zeros((1,4,4,4))
            v = WritableTensorData(data, (3,3,3), (1,1,1))
            self.assertTrue(v.size()   == (1,4,4,4))
            self.assertTrue(v.FoV()    == (3,3,3))
            self.assertTrue(v.offset() == (1,1,1))
            bb = v.bounding_box()
            rg = v.range()
            self.assertTrue(bb == Box((1,1,1),(5,5,5)))
            self.assertTrue(rg == Box((2,2,2),(4,4,4)))

        def testSetPatch(self):
            v = WritableTensorData(np.zeros((1,5,5,5)), (3,3,3), (1,1,1))
            p = np.random.rand(1,3,3,3)
            v.set_patch((4,4,4),p)

    ####################################################################
    unittest.main()

    ####################################################################