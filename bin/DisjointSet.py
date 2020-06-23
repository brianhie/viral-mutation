#! /usr/bin/env python
try:
    from Queue import Queue
except ImportError:
    from queue import Queue
class DisjointSet:
    '''``DisjointSet`` class, implemented using the Up-Tree data structure for amortized O(1) find and union operations'''
    def __init__(self, initial=None):
        '''``DisjointSet`` constructor
        
        Args:
            ``initial`` (iterable): Elements with which to initialize the ``DisjointSet`` (each element will be in its own set)
        '''
        self.parent = dict() # parent[u] = parent of node u
        self.num_below = dict() # num_below[u] = number of nodes below u (including u) (only current for sentinels)
        if initial is not None:
            for x in initial:
                self.add(x)

    def __contains__(self, x):
        '''Check if an element ``x`` exists in this ``DisjointSet``

        Args:
            ``x``: The element to check

        Returns:
            ``bool``: ``True`` if ``x`` exists in this ``DisjointSet``, otherwise ``False``
        '''
        return x in self.parent

    def __iter__(self):
        ''' Iterate over the elements of this ``DisjointSet``'''
        for x in self.parent:
            yield x

    def __len__(self):
        '''Return the number of elements in this ``DisjointSet``

        Returns:
            ``int``: The number of elements contained within this ``DisjointSet``
        '''
        return len(self.parent)

    def __str__(self):
        '''Return a string representation of this ``DisjointSet``

        Returns:
            ``str``: A string representation of this ``DisjointSet``
        '''
        return str(self.sets())

    def add(self, x):
        '''Add a new element ``x`` to this ``DisjointSet`` as a sentinel node

        Args:
            ``x``: The element to insert
        '''
        if x in self:
            raise ValueError("Node already exists: %s"%x)
        self.parent[x] = None; self.num_below[x] = 1

    def remove(self, x):
        '''Remove the element ``x`` from this ``DisjointSet``

        Args:
            ``x``: The element to remove
        '''
        if x not in self:
            raise ValueError("Node not found: %s"%x)
        p = self.parent[x]
        if p is not None:
            p = self.parent[x]; self.num_below[p] -= 1
        for e in self.parent:
            if self.parent[e] == x:
                self.parent[e] = p
        del self.parent[x]; del self.num_below[x]

    def find(self, x):
        '''Return the sentinel node of the element ``x``. Implements path compression along the search

        Args:
            ``x``: The element to find

        Returns:
            The sentinel node of ``x``
        '''
        if x not in self:
            raise ValueError("Node not found: %s"%x)
        explored = Queue(); curr = x
        while self.parent[curr] is not None:
            explored.put(curr); curr = self.parent[curr]
        while not explored.empty():
            self.parent[explored.get()] = curr
        return curr

    def union(self, x, y):
        '''Union the sets containing ``x`` and ``y``. Implements Union-By-Size

        Args:
            ``x``: One of the two elements whose sets will be unioned
            ``y``: One of the two elements whose sets will be unioned
        '''
        if x not in self:
            raise ValueError("Node not found: %s"%x)
        if y not in self:
            raise ValueError("Node not found: %s"%y)
        sx = self.find(x); sy = self.find(y)
        if sx == sy:
            return
        if self.num_below[sx] > self.num_below[sy]:
            self.parent[sy] = sx; self.num_below[sx] += (self.num_below[sy] + 1)
        else:
            self.parent[sx] = sy; self.num_below[sy] += (self.num_below[sx] + 1)

    def sets(self):
        '''Return the sets of this ``DisjointSet``

        Returns:
            ``list`` of ``set``: The sets of this ``DisjointSet``
        '''
        out_sets = dict()
        for x in self.parent:
            p = self.parent[x]
            if p is None:
                p = x
            if p not in out_sets:
                out_sets[p] = set()
            out_sets[p].add(x)
        return list(out_sets.values())
