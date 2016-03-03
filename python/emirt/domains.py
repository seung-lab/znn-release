import numpy as np

class CDisjointSets:
    def __init__(self, N):
        # initally, every voxel is a segment
        # the parent of each voxel
        self._djsets = np.arange(N, dtype='uint32')
        # the rank of each voxel, used for join
        self._rank = np.ones(N, dtype='uint32')
        # total number of voxels
        self._size = N
        # number of sets
        self._sets = N

    def find_root( self, vid ):
        """
        find the segment id of this voxel
        """
        # root id or domain id
        rid = vid
        while ( rid != self._djsets[rid] ):
            rid = self._djsets[rid]

        # path compression
        # current id
        cid = vid
        while (rid!=cid):
            # parent id
            pid = self._djsets[ cid ]
            self._djsets[ cid ] = rid
            cid = pid
        return rid

    def join( self, sid1, sid2 ):
        """
        join two segments
        """
        assert( sid1<self._size and sid2<self._size )
        # if they belong to the same domain
        if sid1==sid2:
            return sid1

        # reduce set number
        self._sets -= 1
        if self._rank[ sid1 ] >= self._rank[ sid2 ]:
            # assign sid1 as the parent of sid2
            self._djsets[ sid2 ] = sid1
            self._rank[ sid1 ] += self._rank[ sid2 ]
            return sid1
        else:
            self._djsets[ sid1 ] = sid2
            self._rank[ sid2 ] += self._rank[ sid1 ]
            return sid2

    def get_seg(self):
        # label all the voxel to root id
        for vid in xrange( self._size ):
            # with path compression,
            # all the voxels will be labeled as root id
            rid = self.find_root( vid )

        return self._djsets

class CDomainLabelSizes:
    """
    the number of voxels
    """
    def __init__(self, lid=None, lsz = 1):
        """
        lid: label id
        vid: voxel id
        """
        # a dictionary containing voxel number of different segment id
        self.sizes = dict()
        if lid:
            self.sizes[lid] = lsz

    def union(self, dm2 ):
        """
        merge with another domain

        Parameters
        ----------
        dm2: CDomain, another domain
        """
        for lid2, sz2 in dm2.sizes.iteritems():
            if self.sizes.has_key(lid2):
                # have common segment id, merge together
                self.sizes[lid2] += sz2
            else:
                # do not have common id, create new one
                self.sizes[lid2] = sz2

    def clear(self):
        """
        delete all the containt
        """
        self.sizes = dict()
        return

    def get_merge_split_errors(self, dm2):
        """
        compute the merge and split error of two domains
        """
        # merging and splitting error
        me = 0
        se = 0
        for lid1, sz1 in self.sizes.iteritems():
            for lid2, sz2 in dm2.sizes.iteritems():
                # ignore the boundaries
                if lid1>0 and lid2>0:
                    if lid1==lid2:
                        # they should be merged together
                        # this is a split error
                        se += sz1 * sz2
                    else:
                        # they should be splitted
                        # this is a merging error
                        me += sz1 * sz2
        return me, se


class CDomains (CDisjointSets):
    """
    the list of watershed domains.
    """
    def __init__( self, lbl ):
        """
        Parameters
        ----------
        lbl: 2D/3D array, manual label image
        """
        assert(lbl.ndim==2 or lbl.ndim==3)

        # disjoint sets, initialize to 0-N-1
        CDisjointSets.__init__(self, lbl.size)

        self.dms = list()
        # voxel id start from 0
        for vid in xrange( lbl.size ):
            # manual labeled segment id
            lid = lbl.flat[vid]
            self.dms.append( CDomainLabelSizes(lid) )
        return

    def find( self, vid ):
        """
        find the corresponding domain of a voxel
        vid: voxel ID
        Return
        ------
        dm: corresponding watershed domain
        """
        rid = self.find_root( vid )
        dm = self.dms[ rid ]
        return rid, dm

    def union(self, vid1, vid2):
        """
        union the two watershed domain of two voxel ids
        """
        # domain id and domain
        rid1, dm1 = self.find(vid1)
        rid2, dm2 = self.find(vid2)

        if rid1 != rid2:
            # compute error
            me, se = dm1.get_merge_split_errors( dm2 )

            # attach the small one to big one
            if self._rank[rid1] < self._rank[rid2]:
                rid1, rid2 = rid2, rid1
                dm1, dm2 = dm2, dm1

            # merge these two domains
            dm1.union(dm2)
            self.dms[rid1] = dm1
            self.dms[rid2].clear()

            # join the sets
            self.join( rid1, rid2 )
            return me, se
        else:
            return 0,0