#!/usr/bin/env python
__doc__ = """

3D Box class.

Modified the code from
https://wiki.python.org/moin/PointsAndRectangles

This code is in the public domain.

Box -- two vectors, forming a box

Kisuk Lee <kisuklee@mit.edu>, 2015
"""

from vector import Vec3d, minimum, maximum

import math

class Box:
    """
    A 3D box identified by two vectors.

    set_coords       -- reset box coordinates
    contains         -- is a point inside?
    overlaps         -- does a box overlap?
    intersect        -- intersection between two boxes, if exist
    merge            -- merge two boxes
    min              -- get min corner point vector
    max              -- get max corner point vector
    expand_by        -- in-place grow (or shrink)
    expanded_by      -- grow (or shrink)

    """

    def __init__(self, v1_or_box, v2=None):
        """Initialize a box from another box or two vectors."""
        if v2 == None:
            self.set_coords(v1_or_box.min(), v1_or_box.max())
        else:
            self.set_coords(Vec3d(v1_or_box), Vec3d(v2))

    def set_coords(self, v1, v2):
        """Reset the box coordinates."""
        self._min  = minimum(v1, v2)
        self._max  = maximum(v1, v2)
        self._size = self._max - self._min

    def size(self):
        return Vec3d(self._size)

    def contains(self, v):
        """Return true if a point is inside the box."""
        (x,y,z) = Vec3d(v)
        return (self._min.x <= x < self._max.x and
                self._min.y <= y < self._max.y and
                self._min.z <= z < self._max.z)

    def overlaps(self, other):
        """Return true if a box overlaps this box."""
        return (self._max.x > other._min.x and self._min.x < other._max.x and
                self._max.y > other._min.y and self._min.y < other._max.y and
                self._max.z > other._min.z and self._min.z < other._max.z)

    def intersect(self, other):
        """Return intersection between this and other box, if overlaps."""
        if self.overlaps(other):
            # min/max corners
            vmin = maximum(self._min, other._min)
            vmax = minimum(self._max, other._max)

            return Box(vmin,vmax)
        else:
            return None

    def merge(self, other):
        """Return merge of this and other box. Two boxes need not overlap."""
       # min/max corners
        vmin = minimum(self._min, other._min)
        vmax = maximum(self._max, other._max)

        return Box(vmin,vmax)

    def translate(self, v):
        self._min += v
        self._max += v

    def min(self):
        """Return the min corner as a vector."""
        return Vec3d(self._min)

    def max(self):
        """Return the max corner as a vector."""
        return Vec3d(self._max)

    def expand_by(self, v):
        """In-place expansion by v."""
        self._min -= v
        self._max += v
        assert(self._min.x < self._max.x and
               self._min.y < self._max.y and
               self._min.z < self._max.z)

    def expanded_by(self, v):
        """Return a box with extended borders."""
        v1 = self._min - v
        v2 = self._max + v
        assert v1.x < v2.x and v1.y < v2.y and v1.z < v2.z
        return Box(v1,v2)

    # Comparison
    def __eq__(self, b):
        return self._min == b.min() and self._max == b.max()

    def __ne__(self, b):
        return not(self == b)

    # String representaion (for printing and debugging)
    def __str__( self ):
        return "<Box (%s)-(%s)>"%(self._min,self._max)

    def __repr__(self):
        return "%s(%r, %r)" % (self.__class__.__name__,self._min,self._max)

# helper box functions
def centered_box(c, s):
    """Return a box of size s centered on c."""
    center = Vec3d(c)
    size   = Vec3d(s)
    half   = size/2
    assert size.x >= 0 and size.y >= 0 and size.z >= 0
    v1 = center - half
    v2 = v1 + size
    return Box(v1,v2)

########################################################################
## Unit Testing                                                       ##
########################################################################
if __name__ == "__main__":

    import unittest

    ####################################################################
    class UnitTestBox(unittest.TestCase):

        def setUp(self):
            pass

        def testCreationAndAccess(self):
            b = Box((1,1,1),[2,2,2])
            self.assertTrue(b.min()  == Vec3d(1,1,1))
            self.assertTrue(b.max()  == Vec3d(2,2,2))
            self.assertTrue(b.size() == Vec3d(1,1,1))
            b = Box((1,0,1),[0,1,0])
            self.assertTrue(b.min()  == Vec3d(0,0,0))
            self.assertTrue(b.max()  == Vec3d(1,1,1))
            self.assertTrue(b.size() == Vec3d(1,1,1))
            b1 = Box(b)
            self.assertTrue(b1.min()  == Vec3d(0,0,0))
            self.assertTrue(b1.max()  == Vec3d(1,1,1))
            self.assertTrue(b1.size() == Vec3d(1,1,1))

        def testComparison(self):
            b1 = Box((1,1,1),[2,2,2])
            b2 = Box((2,2,2),[1,1,1])
            self.assertTrue(b1 == b2)
            b3 = Box((1,1,1),[3,3,3])
            self.assertTrue(b1 != b3)

        def testTranslate(self):
            b = Box((1,1,1),[2,2,2])
            b.translate((-1,-1,-1))
            self.assertTrue(b == Box((0,0,0),(1,1,1)))

        def testContains(self):
            b = Box(Vec3d(1,1,1),[3,3,3])
            self.assertTrue(b.contains(b.min()))
            self.assertTrue(not b.contains(b.max()))
            self.assertTrue(b.contains((2,2,2)))
            self.assertTrue(not b.contains((-2,-2,-2)))

        def testOverlaps(self):
            b1 = Box(Vec3d(1,1,1),[2,2,2])
            b2 = Box(Vec3d(3,3,3),[2,2,2])
            self.assertTrue(b1.overlaps(b1) and b2.overlaps(b2))
            self.assertTrue((not b1.overlaps(b2)) and (not b2.overlaps(b1)))
            b2.translate((-0.5,-0.5,-0.5))
            self.assertTrue(b1.overlaps(b2) and b2.overlaps(b1))

        def testIntersect(self):
            b1 = Box(Vec3d(1,1,1),[2,2,2])
            b2 = Box(Vec3d(3,3,3),[2,2,2])
            self.assertTrue(b1.intersect(b2) == None)
            b2.translate((-0.5,-0.5,-0.5))
            b3 = b1.intersect(b2)
            self.assertTrue(b3 == Box((1.5,1.5,1.5),(2,2,2)))

        def testMerge(self):
            b1 = Box(Vec3d(1,1,1),[2,2,2])
            b2 = Box(Vec3d(3,3,3),[2,2,2])
            b3 = b1.merge(b2)
            self.assertTrue(b3 == Box((1,1,1),(3,3,3)))
            b4 = b3.merge(Box((0,0,0),(-1,-1,-1)))
            self.assertTrue(b4 == Box((-1,-1,-1),(3,3,3)))

        def testExpand(self):
            b1 = Box(Vec3d(1,1,1),[2,2,2])
            b1.expand_by(1)
            self.assertTrue(b1 == Box((0,0,0),(3,3,3)))
            b1.expand_by((-1,0,0))
            self.assertTrue(b1 == Box((1,0,0),(2,3,3)))

        def testCenteredBox(self):
            b = centered_box((0,0,0),(2,2,2))
            self.assertTrue(b == Box((-1,-1,-1),(1,1,1)))
            b = centered_box((0,0,0),(3,3,3))
            self.assertTrue(b == Box((-1,-1,-1),(2,2,2)))

    ####################################################################
    unittest.main()

    ####################################################################