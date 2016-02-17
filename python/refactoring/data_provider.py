#!/usr/bin/env python
__doc__ = """

DaraProvider interface.

Kisuk Lee <kisuklee@mit.edu>, 2015
"""

class DataProvider:
    """
    DataProvider interface.

    Methods
    -------
    next_sample     -- next sample in a predetermined sequence.
    random_sample   -- next sample in a random sequence.

    """

    def __init__(self, spec):
        """
        Initialize DataProvider.
        spec contains every information needed for initialization.
        """
        pass

    def next_sample(self):
        pass

    def random_sample(self):
        pass