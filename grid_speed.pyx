# distutils: language = c++
# cython: language_level=3, boundscheck=False
# cython: language_level=3, wraparound=False

cimport numpy
import numpy
cimport cython
from cython.view cimport array as cvarray
from libcpp.deque cimport deque
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libcpp cimport bool as bool_t


cpdef c_route(numpy.ndarray[unsigned char, ndim=2] directions, 
              numpy.ndarray[numpy.int64_t, ndim=2] world, 
              numpy.int64_t row,
              numpy.int64_t column):
    if directions[row, column] == 32:
        raise ValueError('Unable to find a route.')
    cdef numpy.int64_t i, j
    cdef unsigned char direction
    cdef deque[pair[numpy.int64_t, numpy.int64_t]] fast_queue
    i = row
    j = column

    direction = 32
    first = True
    while directions[i, j] != 43:
        if world[i, j] == 1 or first:
            first = False
            fast_queue.push_back(pair[numpy.int64_t, numpy.int64_t](i, j))
            direction = directions[i, j]
        if direction == 62:
            j += 1
        elif direction == 118:
            i += 1
        elif direction == 60:
            j -= 1
        elif direction == 94:
            i -= 1
        else:
            break
    if directions[i, j] == 43:
        fast_queue.push_back(pair[numpy.int64_t, numpy.int64_t](i, j))
    fast_queue.pop_front()

    return [(i, j) for i, j in fast_queue]

cpdef c_resolve_stations(
                         numpy.ndarray[numpy.int64_t, ndim=2] world,
                         numpy.ndarray[numpy.int64_t, ndim=2] distances,
                         numpy.ndarray[unsigned char, ndim=2] directions):
    safe_stations = numpy.where(world == 2)
    cdef deque[pair[numpy.int64_t, numpy.int64_t]] transport_stations
    cdef pair[numpy.int64_t, numpy.int64_t] station
    cdef numpy.int64_t station_len = safe_stations[0].size
    cdef numpy.int64_t value, i, i_ind, j_ind
    cdef numpy.ndarray[numpy.int64_t] station_i = safe_stations[0]
    cdef numpy.ndarray[numpy.int64_t] station_j = safe_stations[1]
    cdef numpy.int64_t w_val

    for i in range(station_len):
        i_ind = station_i[i]
        j_ind = station_j[i]
        distances[i_ind, j_ind] = 0
        directions[i_ind, j_ind] = 43
        for moving_index in range(j_ind + 1, world.shape[1]):
            w_val = world[i_ind, moving_index]
            if w_val == 0:
                if distances[i_ind, moving_index] >= 1 or distances[i_ind, moving_index] == -1:
                    distances[i_ind, moving_index] = 1
                    directions[i_ind, moving_index] = 60
                continue
            elif w_val == 1:
                if distances[i_ind, moving_index] > 1 or distances[i_ind, moving_index] == -1:
                    distances[i_ind, moving_index] = 1
                    directions[i_ind, moving_index] = 60
                    transport_stations.push_back(pair[numpy.int64_t, numpy.int64_t](i_ind, moving_index))
                break
            elif w_val == 2:
                break
                    
            distances[i_ind, moving_index] = -1
            directions[i_ind, moving_index] = 32
            break

        for moving_index in range(i_ind + 1, world.shape[0]):
            w_val = world[moving_index, j_ind]
            if w_val == 0:
                if distances[moving_index, j_ind] >= 1 or distances[moving_index, j_ind] == -1:
                    distances[moving_index, j_ind] = 1
                    directions[moving_index, j_ind] = 94
                continue
            elif w_val == 1:
                if distances[moving_index, j_ind] > 1 or distances[moving_index, j_ind] == -1:
                    distances[moving_index, j_ind] = 1
                    directions[moving_index, j_ind] = 94
                    transport_stations.push_back(pair[numpy.int64_t, numpy.int64_t](moving_index, j_ind))
                break
            elif w_val == 2:
                break
                    
            distances[moving_index, j_ind] = -1
            directions[moving_index, j_ind] = 32
            break

        for moving_index in range(j_ind - 1, -1, -1):
            w_val = world[i_ind, moving_index]
            if w_val == 0:
                if distances[i_ind, moving_index] >= 1 or distances[i_ind, moving_index] == -1:
                    distances[i_ind, moving_index] = 1
                    directions[i_ind, moving_index] = 62
                continue
            elif w_val == 1:
                if distances[i_ind, moving_index] > 1 or distances[i_ind, moving_index] == -1:
                    distances[i_ind, moving_index] = 1
                    directions[i_ind, moving_index] = 62
                    transport_stations.push_back(pair[numpy.int64_t, numpy.int64_t](i_ind, moving_index))
                break
            elif w_val == 2:
                break
                    
            distances[i_ind, moving_index] = -1
            directions[i_ind, moving_index] = 32
            break

        for moving_index in range(i_ind - 1, -1, -1):
            w_val = world[moving_index, j_ind]
            if w_val == 0:
                if distances[moving_index, j_ind] >= 1 or distances[moving_index, j_ind] == -1:
                    distances[moving_index, j_ind] = 1
                    directions[moving_index, j_ind] = 118
                continue
            elif w_val == 1:
                if distances[moving_index, j_ind] > 1 or distances[moving_index, j_ind] == -1:
                    distances[moving_index, j_ind] = 1
                    directions[moving_index, j_ind] = 118
                    transport_stations.push_back(pair[numpy.int64_t, numpy.int64_t](moving_index, j_ind))
                break
            elif w_val == 2:
                break
                    
            distances[moving_index, j_ind] = -1
            directions[moving_index, j_ind] = 32
            break
    while transport_stations.size() != 0:
        station = transport_stations.front()
        transport_stations.pop_front()
        i_ind = station.first
        j_ind = station.second
        value = distances[i_ind, j_ind]

        for moving_index in range(j_ind + 1, world.shape[1]):
            w_val = world[i_ind, moving_index]
            if w_val == 0:
                if distances[i_ind, moving_index] >= value + 1 or distances[i_ind, moving_index] == -1:
                    distances[i_ind, moving_index] = value + 1
                    directions[i_ind, moving_index] = 60
                continue
            elif w_val == 1:
                if distances[i_ind, moving_index] > value + 1 or distances[i_ind, moving_index] == -1:
                    distances[i_ind, moving_index] = value + 1
                    directions[i_ind, moving_index] = 60
                    transport_stations.push_back(pair[numpy.int64_t, numpy.int64_t](i_ind, moving_index))
                break
            elif w_val == 2:
                break
                    
            distances[i_ind, moving_index] = -1
            directions[i_ind, moving_index] = 32
            break

        for moving_index in range(i_ind + 1, world.shape[0]):
            w_val = world[moving_index, j_ind]
            if w_val == 0:
                if distances[moving_index, j_ind] >= value + 1 or distances[moving_index, j_ind] == -1:
                    distances[moving_index, j_ind] = value + 1
                    directions[moving_index, j_ind] = 94
                continue
            elif w_val == 1:
                if distances[moving_index, j_ind] > value + 1 or distances[moving_index, j_ind] == -1:
                    distances[moving_index, j_ind] = value + 1
                    directions[moving_index, j_ind] = 94
                    transport_stations.push_back(pair[numpy.int64_t, numpy.int64_t](moving_index, j_ind))
                break
            elif w_val == 2:
                break
                    
            distances[moving_index, j_ind] = -1
            directions[moving_index, j_ind] = 32

            break

        for moving_index in range(j_ind - 1, -1, -1):
            w_val = world[i_ind, moving_index]
            if w_val == 0:
                if distances[i_ind, moving_index] >= value + 1 or distances[i_ind, moving_index] == -1:
                    distances[i_ind, moving_index] = value + 1
                    directions[i_ind, moving_index] = 62
                continue
            elif w_val == 1:
                if distances[i_ind, moving_index] > value + 1 or distances[i_ind, moving_index] == -1:
                    distances[i_ind, moving_index] = value + 1
                    directions[i_ind, moving_index] = 62
                    transport_stations.push_back(pair[numpy.int64_t, numpy.int64_t](i_ind, moving_index))
                break
            elif w_val == 2:
                break
                    
            distances[i_ind, moving_index] = -1
            directions[i_ind, moving_index] = 32
            break

        for moving_index in range(i_ind - 1, -1, -1):
            w_val = world[moving_index, j_ind]
            if w_val == 0:
                if distances[moving_index, j_ind] >= value + 1 or distances[moving_index, j_ind] == -1:
                    distances[moving_index, j_ind] = value + 1
                    directions[moving_index, j_ind] = 118
                continue
            elif w_val == 1:
                if distances[moving_index, j_ind] > value + 1 or distances[moving_index, j_ind] == -1:
                    distances[moving_index, j_ind] = value + 1
                    directions[moving_index, j_ind] = 118
                    transport_stations.push_back(pair[numpy.int64_t, numpy.int64_t](moving_index, j_ind))
                break
            elif w_val == 2:
                break
                    
            distances[moving_index, j_ind] = -1
            directions[moving_index, j_ind] = 32
            break