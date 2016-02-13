#!/usr/bin/env python
__doc__ = """

Read-only/writable TensorData classes.

Kisuk Lee <kisuklee@mit.edu>, 2015
"""

from vector import Vec3d
from box import *

import numpy as np
import math

class TensorData:
    """
    Read-oly tensor data.
    This is not a general tensor, in that 4th dimension is regarded as
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

        # set data
        self._set_data(data)
        self._set_FoV(FoV)
        self._set_offset(offset)

    def get_patch(self, pos):
        """
        Extract a subvolume of size _FoV centered on pos.
        """
        assert self._rg.contains(pos)
        pos -= self._offset # local coordinate
        box  = centered_box(pos, self._FoV)
        v1   = b.min()
        v2   = b.max()
        return np.copy(self._data[:,v1[0]:v2[0],v1[1]:v2[1],v1[2]:v2[2]])

    def get_volume(self):
        return np.copy(self._data)

    def size(self):
        return self._data.shape

    def dimension(self):
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

        # data dimension
        return Vec3d(data.shape[1:])

    def _set_data(self, data):
        # TODO: should copy data?
        self._data = data
        self._dim  = self._check_data(data)

        # update bounding box & range
        self._set_bounding_box()
        self._set_range()

    def _set_FoV(self, FoV):
        """
        Set a nonnegative field of view (FoV), i.e., patch size.
        If FoV is zero, set it to the data volume dimension.
        """
        FoV = Vec3d(FoV)
        if FoV == (0,0,0):
            FoV = Vec3d(self._dim)

        # FoV should be nonnegative and smaller than data
        FoV[0] = min(max(FoV[0],0),self._dim[0])
        FoV[1] = min(max(FoV[1],0),self._dim[1])
        FoV[2] = min(max(FoV[2],0),self._dim[2])

        self._FoV = FoV

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
        """
        # top margin
        top = self._FoV/2
        # bottom margin
        btm = self._FoV - top - (1,1,1)

        v1 = self._offset + top
        v2 = self._offset + self._dim - btm

        self._rg = Box(v1,v2)

    # String representaion (for printing and debugging)
    def __str__( self ):
        return "<TensorData>\nsize = %s\ndim  = %s\nFoV  = %s\noff  = %s\n" % \
               (self.size(), self._dim, self._FoV, self._offset)

class WritableTensorData(TensorData):
    """
    Writable volume data

    """
    def __init__(self, data, FoV=(0,0,0), offset=(0,0,0)):
        TensorData.__init__(self,data,FoV,offset)

    def set_patch(self, pos, vol):
        assert self._rg.contains(pos)
        dim  = self._check_data(vol)
        box  = centered_box(pos, dim)

        # local coordinate
        box.translate(-self._offset)
        v1 = box.min()
        v2 = box.max()
        self._data[:,v1[0]:v2[0],v1[1]:v2[1],v1[2]:v2[2]] = vol


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
            # print v.get_volume()

    ####################################################################
    unittest.main()

    ####################################################################