---
title: Build Clang/LLVM with OpenMP Support for Nvidia GPU Offloading
author: Xin Wu (PC²)
date: 13.01.2020
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

The required packages for this GCC build are:

* nvptx-tools:[^nvptxtools]

[^nvptxtools]: At the time of writing, there is no release of nvptx-tools on
  GitHub. For reproducibility the `HEAD` was checked out explicitly.

* nvptx-newlib:[^nvptxnewlib]

[^nvptxnewlib]: At the time of writing, there is no release of nvptx-newlib on
  GitHub. For reproducibility the `HEAD` was checked out explicitly.

* openacc-gcc-9-branch:[^gcc9]

[^gcc9]: This Git-branch is used for development of OpenACC support and related
  functionality. For more info, see <https://gcc.gnu.org/svn.html>.

It's faster to download these packages from the frontend nodes of OCuLUS at PC².
`download.sh` (in `01_download`) is a convenient script to download these
packages as well as to prepare other setups for our build of GCC with OpenMP for
offloading on GPUs.

# Build and Install Packages

## Load CUDA module

Because the GPU-backend of GCC depends on CUDA, we need to load the CUDA module
on OCuLUS.

```bash
module load system/CUDA/10.1.105
```

## Build `nvptx-tools`, accelerator and host GCC compilers

The build scripts can be found in `02_build`. `build.sh` is a driver script for
`ccsalloc` and `realscript.sh` carries out the real build procedure.

Before running

```bash
ccsalloc build.sh
```

it's necessary to adapt some settings in `realscript.sh`, e.g. `CUDADIR`,
`INSTDIR`, and perhaps `make -j` with an appropriate number of processors,
to your working system.

Now, we're done.
