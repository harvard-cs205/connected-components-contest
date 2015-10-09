cimport cython
cimport numpy as np
import numpy as np

cdef extern from "assert.h":
    void cassert "assert"(bint) nogil  # rename to avoid python's assert

# helper functions
cdef inline np.uint32_t cmax(np.uint32_t a,
                             np.uint32_t b) nogil:
        return (a if (a > b) else b)

cdef inline np.uint32_t cmin(np.uint32_t a,
                             np.uint32_t b) nogil:
        return (a if (a < b) else b)


# chase a node's label to its root, doing path compression along the way.
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.uint32_t find_root(np.uint32_t *labels,
                           np.uint32_t node) nogil:
    while labels[node] != node:
        # invariant - nodes can only be parented by lower indexed nodes
        # cassert(labels[node] < node)
        # cassert(labels[labels[node]] <= labels[node])

        labels[node] = labels[labels[node]]  # grandparent path compression
        node = labels[node]
    return node

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef connected_components(int num_nodes,
                           np.uint32_t[:, ::1] links):
    cdef:
        int count = 0, idx
        np.uint32_t a, b, root_a, root_b
        np.uint32_t [::1] labels_return
        np.uint32_t *labels

    assert links.shape[1] == 2, "links should be Nx2"

    labels_return = np.arange(num_nodes, dtype=np.uint32)
    labels = &(labels_return[0])  # for C-level access

    with nogil:
        # merge linked edges
        for idx in range(links.shape[0]):
            a = links[idx, 0]
            b = links[idx, 1]
            root_a = find_root(labels, a)
            root_b = find_root(labels, b)
            if root_a != root_b:
                labels[cmax(root_a, root_b)] = cmin(root_a, root_b)

        # compress all labels
        for idx in range(num_nodes):
            labels[idx] = find_root(labels, idx)

    return labels_return
