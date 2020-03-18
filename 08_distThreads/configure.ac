AC_PREREQ([2.63])
AC_INIT([distThreads], [1.0], [xinwu@mail.uni-paderborn.de])
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_SRCDIR([src/distThreads.c])
AM_INIT_AUTOMAKE([-Wall -Werror foreign subdir-objects])
##############################################################################80
#
# check CUDA
#
##############################################################################80
AC_ARG_VAR([CUDAINC], [The PATH wherein cuda_runtime.h can be found])
if test -z "${CUDAINC}"; then
  AC_SUBST([CUDAINC], [/usr/local/cuda/include])
fi
AC_ARG_VAR([CUDALIB], [The PATH wherein libcudart.so can be found])
if test -z "${CUDALIB}"; then
  AC_SUBST([CUDALIB], [/usr/local/cuda/lib64])
fi
##############################################################################80
#
# check C compiler
#
##############################################################################80
AC_PROG_CC([clang gcc])
AS_IF([test "${CC}" = gcc],
  [CFLAGS="-Wall -fopenmp -foffload=nvptx-none -I${CUDAINC} $CFLAGS"
   LDFLAGS="-L${CUDALIB} $LDFLAGS"])
AS_IF([test "${CC}" = clang],
  [CFLAGS="-Wall -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -I${CUDAINC} \
   -Xopenmp-target -march=sm_61 $CFLAGS"
   LDFLAGS="-L${CUDALIB} $LDFLAGS"])
##############################################################################80
#
# check archiver
#
##############################################################################80
AC_PROG_RANLIB
AM_PROG_AR
##############################################################################80
#
# check headers
#
##############################################################################80
AC_CHECK_HEADER([cuda_runtime.h], [],
  [AC_MSG_ERROR([cuda_runtime.h required, but not found])], [])
##############################################################################80
#
# check libraries
#
##############################################################################80
AC_CHECK_LIB([cudart], [cudaSetDevice], [],
  [AC_MSG_ERROR([libcudart required, but not found])], [])
##############################################################################80
#
# check Doxygen
#
##############################################################################80
AC_ARG_VAR([DOXYGEN], [Doxygen: source documentation generation program])
AC_CHECK_PROG([DOXYGEN], [doxygen], [doxygen], [], [], [])
AS_IF([test -z "${DOXYGEN}"],
  [AC_MSG_WARN([doxygen not found - continue without doxygen support])])
AM_CONDITIONAL([HAVE_DOXYGEN], [test -n "${DOXYGEN}"])
AM_COND_IF([HAVE_DOXYGEN], [AC_CONFIG_FILES([docs/Doxyfile])])
AM_COND_IF([HAVE_DOXYGEN], [AC_CONFIG_FILES([docs/Makefile])])
##############################################################################80
#
# check ccsalloc (in OpenCCS)
#
##############################################################################80
AC_ARG_VAR([CCSALLOC], [OpenCCS: Open Computing Center Software])
AC_CHECK_PROG([CCSALLOC], [ccsalloc], [ccsalloc], [], [], [])
AS_IF([test -z "${CCSALLOC}"],
  [AC_MSG_WARN([ccsalloc not found - continue without OpenCCS support])])
AM_CONDITIONAL([HAVE_CCSALLOC], [test -n "${CCSALLOC}"])
##############################################################################80
#
# create final files
#
##############################################################################80
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES([Makefile
                 src/Makefile
                 tests/Makefile])
AC_OUTPUT

echo "
//============================================================================80

Configuration:

  CC      : ${CC}
  CFLAGS  : ${CFLAGS}
  LDFLAGS : ${LDFLAGS}

//============================================================================80

Now, type make to build ..."
