---
title: Bugs Found in Clang/LLVM
author: Xin Wu (PC²)
date: 15.01.2020
---

# Activation of Accelerator

`omp_get_num_devices()` constantly returns 0, if accelerator(s) have not been
activated by an OpenMP directive, even though there are accelerator(s) in the
computing system. See `02_dataTransRate`.
