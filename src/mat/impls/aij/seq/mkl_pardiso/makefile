#requirespackage   'PETSC_HAVE_MKL_PARDISO'

ALL: lib

CFLAGS   = ${MKL_PARDISO_INCLUDE}
FFLAGS   =
SOURCEC	 = mkl_pardiso.c
SOURCEF	 =
SOURCEH	 =
LIBBASE	 = libpetscmat
DIRS	 =
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/aij/seq/mkl_pardiso/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
