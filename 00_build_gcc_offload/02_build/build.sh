#!/bin/bash
#CCS -N build
#CCS -t 600m
#CCS -g pc2-mitarbeiter
#CCS --res=rset=1:ncpus=16:mem=32g:vmem=32g:tesla=1

module load system/CUDA/10.1.105
sh -x realscript.sh 2>&1 | tee build.log
