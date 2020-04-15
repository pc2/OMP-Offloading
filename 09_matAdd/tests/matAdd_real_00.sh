#!/bin/bash
#CCS -N matAdd
#CCS -t 600m
#CCS -g pc2-mitarbeiter
#CCS --res=rset=1:gtx1080=1,place=:excl

echo "hallo from $(hostname)"
../src/matAdd $((2**12))
../src/matAdd $((2**13))
