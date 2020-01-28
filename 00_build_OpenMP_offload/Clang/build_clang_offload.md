---
title: Build Clang/LLVM with OpenMP Support for Nvidia GPU Offloading
author: Xin Wu (PC²)
date: 28.01.2020
---

# Check Nvidia GPU

The build procedure was carried out on a Tesla node of OCuLUS at PC². It
features an Nvidia Tesla K20X GPU. Thus it's necessary to check the Tesla K20X
GPU on the compute node, before building Clang with OpenMP support for offloading
computation on Nvidia GPU.

The relevant scripts and log files can be found in `00_check_gpu`.

`tesla.sh` is a driver script and should be submitted with `ccsalloc`:

```bash
ccsalloc testa.sh
```

`realscript.sh` does the real job and the output can be found in `tesla.log`.

# Build Clang and Necessary Toolchains

The necessary toolchains for building Clang need to be built first. For this
purpose we have built GCC 8.3.0,[^gcc830] binutils, autoconf, automake, OpenSSL,
CMake, and ncurses.

[^gcc830]: At the time of writing, GCC 9.2.0 is not supported for building Clang
  with OpenMP offloading to GPU.

After the toolchains have been built, Clang can be built with GCC 8.3.0 by using
the following script:

```bash
pkgname="llvmorg-9.0.1"
curl -L -O https://github.com/llvm/llvm-project/archive/${pkgname}.tar.gz
tar xf ${pkgname}.tar.gz
BUILDIR="GCC"
rm -fr   $BUILDIR
mkdir -p $BUILDIR
cd       $BUILDIR
cmake                                                                          \
  -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;libcxx;libcxxabi;lld;openmp" \
  -DCMAKE_PREFIX_PATH="${TOOLCHAINS}"                                          \
  -DCMAKE_BUILD_TYPE=Release                                                   \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX"                                          \
  -DCMAKE_INSTALL_PREFIX=${DESTDIR}                                            \
  -DLLVM_ENABLE_ASSERTIONS=ON                                                  \
  -DLLVM_ENABLE_BACKTRACES=ON                                                  \
  -DLLVM_ENABLE_WERROR=OFF                                                     \
  -DBUILD_SHARED_LIBS=OFF                                                      \
  -DLLVM_ENABLE_RTTI=ON                                                        \
  -DCLANG_OPENMP_NVPTX_DEFAULT_ARCH=sm_61                                      \
  -DLIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES=35,37,50,52,60,61,70,75            \
  -DCMAKE_C_COMPILER=gcc                                                       \
  -DCMAKE_CXX_COMPILER=g++                                                     \
  -G "Unix Makefiles" ../llvm-project-${pkgname}/llvm 2>&1 | tee ${pkgname}.${BUILDIR}.cmak.logfile
make -j 16                                            2>&1 | tee ${pkgname}.${BUILDIR}.make.logfile
make install                                          2>&1 | tee ${pkgname}.${BUILDIR}.inst.logfile
cd ..
```

# Bootstrap Clang with `libc++`

We need to bootstrap Clang for OpenMP offloading. The following script
bootstraps Clang with its own `libc++`:

```bash
pkgname="llvmorg-9.0.1"
curl -L -O https://github.com/llvm/llvm-project/archive/${pkgname}.tar.gz
tar xf ${pkgname}.tar.gz
BUILDIR="LIBCXX"
rm -fr   $BUILDIR
mkdir -p $BUILDIR
cd       $BUILDIR
cmake                                                                          \
  -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;libcxx;libcxxabi;lld;openmp" \
  -DCMAKE_PREFIX_PATH="${TOOLCHAINS}"                                          \
  -DCMAKE_BUILD_TYPE=Release                                                   \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX"                                          \
  -DCMAKE_INSTALL_PREFIX=${DESTDIR}                                            \
  -DLLVM_ENABLE_ASSERTIONS=ON                                                  \
  -DLLVM_ENABLE_BACKTRACES=ON                                                  \
  -DLLVM_ENABLE_WERROR=OFF                                                     \
  -DBUILD_SHARED_LIBS=OFF                                                      \
  -DLLVM_ENABLE_RTTI=ON                                                        \
  -DCLANG_OPENMP_NVPTX_DEFAULT_ARCH=sm_61                                      \
  -DLIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES=35,37,50,52,60,61,70,75            \
  -DCMAKE_C_COMPILER=clang                                                     \
  -DCMAKE_CXX_COMPILER=clang++                                                 \
  -DCMAKE_CXX_FLAGS="-stdlib=libc++"                                           \
  -DCMAKE_CXX_LINK_FLAGS="-stdlib=libc++"                                      \
  -G "Unix Makefiles" ../llvm-project-${pkgname}/llvm 2>&1 | tee ${pkgname}.${BUILDIR}.cmak.logfile
make -j 16                                            2>&1 | tee ${pkgname}.${BUILDIR}.make.logfile
make install                                          2>&1 | tee ${pkgname}.${BUILDIR}.inst.logfile
cd ..
```

To access this version of Clang on OCuLUS:

```bash
module load clang/9.0.1_BS_libcxx_CUDA10.1
```

# Bootstrap Clang with `libstdc++`

Clang can also be bootstrapped with GNU's `libstdc++` with the following script:

```bash
pkgname="llvmorg-9.0.1"
curl -L -O https://github.com/llvm/llvm-project/archive/${pkgname}.tar.gz
tar xf ${pkgname}.tar.gz
BUILDIR="LIBSTDCXX"
rm -fr   $BUILDIR
mkdir -p $BUILDIR
cd       $BUILDIR
cmake                                                                          \
  -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;libcxx;libcxxabi;lld;openmp" \
  -DCMAKE_PREFIX_PATH="${TOOLCHAINS}"                                          \
  -DCMAKE_BUILD_TYPE=Release                                                   \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX"                                          \
  -DCMAKE_INSTALL_PREFIX=${DESTDIR}                                            \
  -DLLVM_ENABLE_ASSERTIONS=ON                                                  \
  -DLLVM_ENABLE_BACKTRACES=ON                                                  \
  -DLLVM_ENABLE_WERROR=OFF                                                     \
  -DBUILD_SHARED_LIBS=OFF                                                      \
  -DLLVM_ENABLE_RTTI=ON                                                        \
  -DCLANG_OPENMP_NVPTX_DEFAULT_ARCH=sm_61                                      \
  -DLIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES=35,37,50,52,60,61,70,75            \
  -DCMAKE_C_COMPILER=clang                                                     \
  -DCMAKE_CXX_COMPILER=clang++                                                 \
  -G "Unix Makefiles" ../llvm-project-${pkgname}/llvm 2>&1 | tee ${pkgname}.${BUILDIR}.cmak.logfile
make -j 16                                            2>&1 | tee ${pkgname}.${BUILDIR}.make.logfile
make install                                          2>&1 | tee ${pkgname}.${BUILDIR}.inst.logfile
cd ..
```

To access this version of Clang on OCuLUS:

```bash
module load clang/9.0.1_BS_libstdcxx_CUDA10.1
```

