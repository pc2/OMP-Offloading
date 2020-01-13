#!/bin/bash
#
# clean up and copy
#
echo "Copy files ..."
for i in gcc-gcc-9_2_0-release nvptx-newlib nvptx-tools; do
  echo                                                                  $i
  rm  -fr                                                               $i
  cp -afr /scratch/pc2-mitarbeiter/xinwu/GCC_OpenMP_OpenACC/01_download/$i .
done
echo "Finish copy files"
#
# environment variables
#
TARGSYS=$(gcc-gcc-9_2_0-release/config.guess)
CUDADIR=/cm/shared/apps/pc2/EB-SW/software/system/CUDA/10.1.105
##INSTDIR=/scratch/pc2-mitarbeiter/xinwu/GCC_OpenMP_OpenACC/99_gcc9_gpu
INSTDIR=/cm/shared/apps/pc2/GCC/9.2.0-offload
#
# nvptx-tools
#
echo "build nvptx-tools ..."
cd nvptx-tools
./configure                                     \
    --with-cuda-driver-include=$CUDADIR/include \
    --with-cuda-driver-lib=$CUDADIR/lib64       \
    --prefix=$INSTDIR
make
make install
cd ..
echo "Finish build nvptx-tools"
#
# Accel_GCC
#
echo "build Accel_GCC ..."
mkdir Accel_GCC
cd    Accel_GCC
../gcc-gcc-9_2_0-release/configure                  \
    --target=nvptx-none                             \
    --enable-as-accelerator-for=$TARGSYS            \
    --with-build-time-tools=$INSTDIR/nvptx-none/bin \
    --disable-sjlj-exceptions                       \
    --enable-newlib-io-long-long                    \
    --enable-languages="c,c++,fortran,lto"          \
    --prefix=$INSTDIR
make -j16
make install
cd ..
echo "Finish build Accel_GCC"
#
# Host_GCC
#
echo "Host_GCC ..."
mkdir Host_GCC
cd    Host_GCC
../gcc-gcc-9_2_0-release/configure              \
    --enable-offload-targets=nvptx-none         \
    --with-cuda-driver-include=$CUDADIR/include \
    --with-cuda-driver-lib=$CUDADIR/lib64       \
    --disable-bootstrap                         \
    --disable-multilib                          \
    --enable-languages="c,c++,fortran,lto"      \
    --prefix=$INSTDIR
make -j16
make install
cd ..
echo "Finish Host_GCC"
#
# Done
#
echo "Done"
