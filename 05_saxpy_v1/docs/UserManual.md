---
title: saxpy
author: Xin Wu (PC²)
date: 10.01.2020
---

# Introduction

`saxpy` performs the `axpy` operation on host as well as accelerator and then
compares the FLOPS performance.

The `axpy` operation is defined as:

$$ y := a * x + y $$

where:

* `a` is a scalar.
* `x` and `y` are vectors each with n elements.

The initial value of `a` and elements of `x[]` and `y[]` are specially designed,
so that the floating-point calculations on host and accelerator can be compared
_exactly_.

Please note that only _one GPU thread_ is used for the `axpy` calculation on
accelerator in this version. This can be verified by uncomment the `CFLAGS` line
in `configure.ac`.

# Usage

```bash
saxpy
```

