#!/usr/bin/env python
"""Spacegrid route calculator using reverse BFS."""
import numpy
import grid_speed


class ResultStruct:
    """Encapsulates the result of the BFS as well as all the data structures."""

    FREE_SPACE = 0
    TRANSPORT_SPACE = 1
    SAFE_SPACE = 2

    def __init__(self, grid):
        if not isinstance(grid, numpy.ndarray):
            raise TypeError('Only Numpy arrays are supported.')
        if grid.ndim != 2:
            raise TypeError('Only 2D arrays are supported.')
        if not numpy.issubdtype(grid.dtype, numpy.integer):
            raise TypeError('Only integer arrays are supported.')

        self.distances = numpy.full(grid.shape, -1, dtype=int)
        self.world = grid.copy().astype(int)
        self.c_directions = numpy.full(grid.shape, 32, dtype=numpy.ubyte)
        self.directions = self.c_directions.view('c')
        self.safe_factor = 0.0

    def route(self, row, column):
        """Returns list with the best route for given cell."""
        r = 0
        c = 0
        try:
            r = int(row)
            c = int(column)
        except ValueError:
            raise IndexError

        if r > self.c_directions.size - 1 or c > self.c_directions.size - 1:
            raise IndexError
        return grid_speed.c_route(self.c_directions, self.world, r, c)

    def calc_routes(self):
        """Calculates best route for the full graph using reverse BFS."""
        grid_speed.c_resolve_stations(self.world, self.distances, self.c_directions)
        self._post_calc()

    def _post_calc(self):
        if self.distances.size == 0:
            self.safe_factor = numpy.nan
        else:
            self.safe_factor = numpy.count_nonzero(self.distances >= 0) / self.distances.size

def escape_routes(grid):
    """Returns directions, distances and route method for given graph."""

    result = ResultStruct(grid)
    result.calc_routes()
    return result
