#!/bin/bash
#CCS -N matMul
#CCS -t 10m
#CCS -g pc2-mitarbeiter
#CCS --res=rset=1:gtx1080=1,place=:excl

../src/matMul $((2**12))
