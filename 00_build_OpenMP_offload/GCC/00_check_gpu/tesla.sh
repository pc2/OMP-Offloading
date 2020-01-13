#!/bin/bash
#CCS -N nvidia_smi
#CCS -t 1m
#CCS -g pc2-mitarbeiter
#CCS --res=rset=1:ncpus=1:mem=8g:vmem=16g:tesla=1

sh -x realscript.sh 2>&1 | tee tesla.log
