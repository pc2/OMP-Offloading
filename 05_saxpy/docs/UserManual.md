---
title: saxpy
author: Xin Wu (PC²)
date: 05.04.2020
---

# Introduction

`saxpy` performs the `saxpy` operation on host as well as accelerator.
The performance (in MB/s) for different implementations is also compared.

The `saxpy` operation is defined as:

$$ y := a * x + y $$

where:

* `a` is a scalar.
* `x` and `y` are single-precision vectors each with n elements.
* For testing n is assumed to be $2^{22}$.
* The following table only summarizes the most important points. For more
  details on the ial-th implementation see comments in `hsaxpy.c` (on host)
  and `asaxpy.c` (on accelerator).

    - on host

| ial |  Remarks                                                               |
|:---:|------------------------------------------------------------------------|
|  0  | naive implementation                                                   |
|  1  | saxpy in MKL                                                           |

    - on accl

| ial |  Remarks                                                               |
|:---:|------------------------------------------------------------------------|
|  0  | <<<             1,   1>>>, TOO SLOW! not tested                        |
|  1  | <<<             1, 128>>>                                              |
|  2  | <<<           128,   1>>>                                              |
|  3  | <<<           128, 128>>>                                              |
|  4  | <<<n /        128, 128>>>                                              |
|  5  | <<<n / (128 * 16), 128>>>, 16x loop unrolling                          |
|  6  | cublasSaxpy in CUBLAS                                                  |

# Usage

```bash
saxpy
```

