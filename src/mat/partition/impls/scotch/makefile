ALL: lib

#requirespackage 'PETSC_HAVE_PTSCOTCH'

CFLAGS   =
FFLAGS   =
CPPFLAGS = ${SCOTCH_INCLUDE}
SOURCEC  = scotch.c
SOURCEH  =
LIBBASE  = libpetscmat
LOCDIR   = src/mat/partition/impls/scotch/
MANSEC   = MatOrderings

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
