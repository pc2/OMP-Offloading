#!/bin/bash
#
# This script builds clang/llvm with OpenMP support for offloading to accelerator
# by using GCC compilers.
#
# 1. check gcc and g++ versions
#
gcc --version
g++ --version
#
# 2. create llvmdirname
#
buildat=$(date "+%d%m%Y")
rm -fr llvm-project
tar xf llvm-project.tar.xz
cd     llvm-project
githead=$(git rev-parse --short HEAD)
cd     ..
llvmdirname=${buildat}_git${githead}_gcc
echo $llvmdirname
sleep 5
#
# cmake and make
#
rm -fr buildWithGCC
mkdir  buildWithGCC
cd     buildWithGCC
cmake                                                   \
  -DLLVM_ENABLE_PROJECTS="clang;openmp"                 \
  -DCMAKE_PREFIX_PATH=/opt/z3/4.8.7                     \
  -DCMAKE_BUILD_TYPE=Release                            \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX"                   \
  -DCMAKE_INSTALL_PREFIX=/opt/llvm/${llvmdirname}       \
  -DLLVM_ENABLE_ASSERTIONS=ON                           \
  -DLLVM_ENABLE_BACKTRACES=ON                           \
  -DLLVM_ENABLE_WERROR=OFF                              \
  -DBUILD_SHARED_LIBS=OFF                               \
  -DLLVM_ENABLE_RTTI=ON                                 \
  -DCLANG_OPENMP_NVPTX_DEFAULT_ARCH=sm_61               \
  -DLIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES=35,60,61,70 \
  -DCMAKE_C_COMPILER=gcc                                \
  -DCMAKE_CXX_COMPILER=g++                              \
  -G "Unix Makefiles" ../llvm-project/llvm 2>&1 | tee       cmake.logfile
make -j4                                   2>&1 | tee        make.logfile
sudo make install                          2>&1 | tee makeinstall.logfile
