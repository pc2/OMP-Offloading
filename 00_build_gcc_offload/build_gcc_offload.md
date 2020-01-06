---
title: Build GCC with OpenMP Support for Nvidia GPU Offloading
author: Xin Wu (PC²)
date: 06.01.2020
---

# Check Nvidia GPU

The build procedure was carried out on a Tesla node of OCuLUS at PC². It
features an Nvidia Tesla K20X GPU. Thus it's necessary to check the Tesla K20X
GPU on the compute node, before building GCC with OpenMP support for offloading
computation on Nvidia GPU.

The relevant scripts and log files can be found in `00_check_gpu`.

`tesla.sh` is a driver script and should be submitted with `ccsalloc`:

```bash
ccsalloc testa.sh
```

`realscript.sh` does the real job and the output can be found in `tesla.log`.

# Download Packages and Preparation

# Build and Install Packages

