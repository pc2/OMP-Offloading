#!/bin/bash
#CCS -N taskwait
#CCS -t 10m
#CCS -g pc2-mitarbeiter
#CCS --res=rset=1:ncpus=1:mem=4g:vmem=8g:gtx1080=2

if [ 0 -eq 1 ]; then
#
# Asynchronous offloading is not available in GCC 9.2.0.
#
notImpld=$(../src/taskwait 2>&1 | grep "GOMP_OFFLOAD_async_run")
[[ $notImpld =~ "unimplemented" ]]
else
#
# Asynchronous offloading is     available in Clang/LLVM 9.0.1.
#
../src/taskwait
fi
