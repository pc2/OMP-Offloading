#!/bin/bash
#
# nvptx-tools
#
echo "nvptx-tools"
git clone https://github.com/MentorEmbedded/nvptx-tools.git
cd                                          nvptx-tools
git checkout -b gcc9_gpu 5f6f343a302d620b0868edab376c00b15741e39e
cd ..
#
# nvptx-newlib
#
echo "nvptx-newlib"
git clone https://github.com/MentorEmbedded/nvptx-newlib.git
cd                                          nvptx-newlib
git checkout -b gcc9_gpu 66dd175a9d3aea387715f00ff18ef7e535cd1272
cd ..
#
# openacc-gcc-9-branch
#
echo "openacc-gcc-9-branch"
wget https://github.com/gcc-mirror/gcc/archive/gcc-9_2_0-release.tar.gz
tar xf                                         gcc-9_2_0-release.tar.gz
cd                                         gcc-gcc-9_2_0-release
./contrib/download_prerequisites
ln -s ../nvptx-newlib/newlib newlib
cd ..
#
# Done
#
echo "Done"
