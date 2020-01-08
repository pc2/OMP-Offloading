#!/bin/bash
#CCS -N dataTransRate
#CCS -t 10m
#CCS -g pc2-mitarbeiter
#CCS --res=rset=1:ncpus=1:mem=4g:vmem=8g:gtx1080=2

notImpld=$(../src/taskwait 2>&1 | grep "GOMP_OFFLOAD_async_run")
[[ $notImpld =~ "unimplemented" ]]
