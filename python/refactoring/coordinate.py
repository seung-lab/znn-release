#!/usr/bin/env python
__doc__ = """

3D Point and Box classes.

Modified the code from
https://wiki.python.org/moin/PointsAndRectangles

This code is in the public domain.

Point   -- point with (x,y,z) coordinates
Box     -- two points, forming a box

Kisuk Lee <kisuklee@mit.edu>, 2015
"""

import math


class Point:

    """A point identified by (x,y,z) coordinates.

    supports: +, -, *, /, str, repr

    length        -- calculate length of vector to point from origin
    distance_to   -- calculate distance between two points
    as_tuple      -- construct tuple (x,y,z)
    clone         -- construct a duplicate
    integerize    -- convert x & y & z to integers
    floatize      -- convert x & y & z to floats
    move_to       -- reset x & y & z
    translate     -- move (in place) +dx, +dy, +dz as spec'd by point
    translate_xyz -- move (in place) +dx, +dy, +dz
    """

    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, p):
        """Point(x1+x2, y1+y2, z1+z2)"""
        return Point(self.x+p.x, self.y+p.y, self.z+p.z)

    def __sub__(self, p):
        """Point(x1-x2, y1-y2, z1-z2)"""
        return Point(self.x-p.x, self.y-p.y, self.z-p.z)

    def __mul__( self, scalar ):
        """Point(x*c, y*c, z*c)"""
        return Point(self.x*scalar, self.y*scalar, self.z*scalar)

    def __div__(self, scalar):
        """Point(x/c, y/c, z/c)"""
        return Point(self.x/scalar, self.y/scalar, self.z/scalar)

    def __str__(self):
        return "(%s, %s, %s)" % (self.x, self.y, self.z)

    def __repr__(self):
        return "%s(%r, %r, %r)" % (self.__class__.__name__,
                                   self.x, self.y, self.z)

    def length(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def distance_to(self, p):
        """Calculate the distance between two points."""
        return (self - p).length()

    def as_tuple(self):
        """(x, y, z)"""
        return (self.x, self.y, self.z)

    def clone(self):
        """Return a full copy of this point."""
        return Point(self.x, self.y, self.z)

    def integerize(self):
        """Convert co-ordinate values to integers."""
        self.x = int(self.x)
        self.y = int(self.y)
        self.z = int(self.z)

    def floatize(self):
        """Convert co-ordinate values to floats."""
        self.x = float(self.x)
        self.y = float(self.y)
        self.z = float(self.z)

    def move_to(self, x, y, z):
        """Reset x & y & z coordinates."""
        self.x = x
        self.y = y
        self.z = z

    def translate(self, p):
        '''Translate to new (x+dx,y+dy,z+dz).'''
        self.x = self.x + p.x
        self.y = self.y + p.y
        self.z = self.y + p.z

    def translate_xyz(self, dx, dy, dz):
        '''Move to new (x+dx,y+dy,z+dz).'''
        self.x = self.x + dx
        self.y = self.y + dy
        self.z = self.y + dz

    def centered_box(self, sx, sy, sz):
        '''Return a Box centered on this point'''
        size = Point(sx, sy, sz)
        half = size/2
        pmin = self - half
        pmax = pmin + size
        return Box(pmin, pmax)


class Box:

    """A 3D box identified by two points.

    set_points       -- reset box coordinates
    contains         -- is a point inside?
    overlaps         -- does a box overlap?
    intersect        -- intersection between two boxes, if exist
    merge            -- merge two boxes
    min_corner       -- get min corner
    max_corner       -- get max corner
    expanded_by      -- grow (or shrink)
    expanded_by_xyz  -- grow (or shrink) by dx, dy, dz
    """

    def __init__(self, pt1, pt2):
        """Initialize a box from two points."""
        self.set_points(pt1, pt2)

    def set_points(self, pt1, pt2):
        """Reset the box coordinates."""
        (x1, y1, z1) = pt1.as_tuple()
        (x2, y2, z2) = pt2.as_tuple()

        xmin = min(x1, x2)
        xmax = max(x1, x2)
        ymin = min(y1, y2)
        ymax = max(y1, y2)
        zmin = min(z1, z2)
        zmax = max(z1, y2)

        self.min  = Point(xmin, ymin, zmin)
        self.max  = Point(xmax, ymax, zmax)
        self.size = (xmax-xmin, ymax-ymin, zmax-zmin)

    def contains(self, pt):
        """Return true if a point is inside the box."""
        (x,y,z) = pt.as_tuple()
        return (self.xmin <= x <= self.xmax and
                self.ymin <= y <= self.ymax and
                self.zmin <= z <= self.zmax)

    def overlaps(self, other):
        """Return true if a box overlaps this box."""
        return (self.xmax > other.xmin and self.xmin < other.xmax and
                self.ymax < other.ymin and self.ymin > other.ymax and
                self.zmax < other.zmin and self.zmin > other.zmax)

    def intersect(self, other):
        """Return intersection between this and other box, if overlaps"""
        if self.overlaps(other):
            # min corner
            xmin = max(self.xmin, other.xmin)
            ymin = max(self.ymin, other.ymin)
            zmin = max(self.zmin, other.zmin)

            # max corner
            xmax = min(self.xmax, other.xmax)
            ymax = min(self.ymax, other.ymax)
            zmax = min(self.zmax, other.zmax)

            # min/max corners
            pt1  = Point(xmin,ymin,zmin)
            pt2  = Point(xmax,ymax,zmax)

            return Box(pt1, pt2)
        else:
            return None

    def merge(self, other):
        """Return merge of this and other box.
        Two boxes need not overlap"""
        # min corner
        xmin = min(self.xmin, other.xmin)
        ymin = min(self.ymin, other.ymin)
        zmin = min(self.zmin, other.zmin)

        # max corner
        xmax = max(self.xmax, other.xmax)
        ymax = max(self.ymax, other.ymax)
        zmax = max(self.zmax, other.zmax)

        # min/max corners
        pt1  = Point(xmin,ymin,zmin)
        pt2  = Point(xmax,ymax,zmax)

        return Box(pt1, pt2)

    def min_corner(self):
        """Return the min corner as a Point."""
        return Point(self.xmin, self.ymin, self.zmin)

    def max_corner(self):
        """Return the max corner as a Point."""
        return Point(self.xmax, self.ymax, self.zmax)

    def expanded_by(self, n):
        """Return a box with extended borders.

        Create a new box that is wider and taller than the
        immediate one. All sides are extended by "n" points.
        """
        p1 = Point(self.xmin-n, self.ymin-n, self.zmin-n)
        p2 = Point(self.xmax+n, self.ymax+n, self.zmax+n)
        return Box(p1, p2)

    def expanded_by_xyz(self, dx, dy, dz):
        """Return a box with extended borders."""
        p1 = Point(self.xmin-dx, self.ymin-dy, self.zmin-dz)
        p2 = Point(self.xmax+dx, self.ymax+dy, self.zmax+dz)
        return Box(p1, p2)

    def __str__( self ):
        return "<Box (%s,%s,%s)-(%s,%s,%s)>" %
               (self.xmin,self.ymin,self.zmin,
                self.xmax,self.ymax,self.zmax)

    def __repr__(self):
        return "%s(%r, %r)" % (self.__class__.__name__,
                               Point(self.xmin, self.ymin, self.zmin),
                               Point(self.xmax, self.ymax, self.zmax))