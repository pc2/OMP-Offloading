#!/bin/bash
#CCS -N matMul
#CCS -t 600m
#CCS -g pc2-mitarbeiter
#CCS --res=rset=1:gtx1080=1,place=:excl

echo "hallo from $(hostname)"
../src/matMul $((2**12))
