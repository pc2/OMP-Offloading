#!/bin/bash
#CCS -N saxpy
#CCS -t 10m
#CCS -g pc2-mitarbeiter
#CCS --res=rset=1:ncpus=1:mem=4g:vmem=8g:tesla=1

ngth=1
while [ $ngth -le 1024 ]; do
  ../src/saxpy $ngth
  (( ngth *= 2 ))
done
