# utils.h python interface

# Referencing the utils header file
cdef extern from 'cpp/src/utils.h' namespace 'utils':
    void initTables()
    void freeTables()

    # forwarding log and sigmoid method directly
    cpdef float log(float)
    cpdef float sigmoid(float)

# aliasing the method
cdef inline void init_tables():
    initTables()

cdef inline void free_tables():
    freeTables()
