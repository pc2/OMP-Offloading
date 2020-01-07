# Introduction

# List of Projects

* 00_build_gcc_offload

  Documentation and scripts for building GCC with OpenMP support for
  Nvidia GPU offloading

* 01_accelQuery

  `accelQuery` searches accelerator(s) on a heterogeneous computer.
  Accelerator(s), if found, will be enumerated with some basic info.

  The directory tree:

  - src: source code
  - docs: documentation
  - tests: some tests

* 02_dataTransRate

  `dataTransRate` gives the data transfer rate (in MB/sec) from `src` to `dst`.

  The possible situations are:

  * h2h: `src` = host  and `dst` = host
  * h2a: `src` = host  and `dst` = accel
  * a2a: `src` = accel and `dst` = accel

  The directory tree:

  - src: source code
  - docs: documentation
  - tests: some tests
