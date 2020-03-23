#!/bin/bash
#CCS -N dataTransRate
#CCS -t 10m
#CCS -g pc2-mitarbeiter
#CCS --res=rset=1:gtx1080=1,place=:excl

echo "hallo from $(hostname)"
../src/dataTransRate
