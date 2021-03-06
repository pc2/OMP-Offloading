---
title: matMul
author: Xin Wu (PC²)
date: 19.03.2020
---

# Introduction

`matMul` performs matrix multiplication in single-precision on GPU. The
performance (in GFLOPS) for different implementations is compared and the
numerical results are also verified.

* Column-major is assumed thru the entire code!

* `i` and `j` are indices for row and column, respectively.

* For testing the dimension of all matrices are assumed to be 4096 x 4096.

* The following table only summarizes the most important points. For more
  details on the ial-th OpenMP GPU implementation see comments in `matMulAB.c`.

| ial |  Remarks                                                               |
|:---:|------------------------------------------------------------------------|
|  0  | jik-loop, 2^9 threads * 2^3 teams,                                     |
|     | uncoalesced memory access                                              |
|  1  | jki-loop, 2^9 threads * 2^3 teams,                                     |
|     | uncoalesced memory access, uncoalesced r&w in innermost loop           |
|  2  | jik-loop, 2^9 threads * 2^f teams, collapse(2)                         |
|  3  | jki-loop, 2^9 threads * 2^f teams, collapse(2),                        |
|     | race condition for writing c!                                          |
|  4  | jik-loop, 2^9 threads * 2^f teams, collapse(2),                        |
|     | 4x k-loop unrolling                                                    |
|  5  | jik-loop, 2^7 threads * 2^f teams, collapse(3),                        |
|     | 4x i-loop unrolling (stride of 2^7 rows),                              |
|     | 4x k-loop unrolling,                                                   |
|     | rb: 4x data reuse                                                      |
|  6  | jik-loop, 2^7 threads * 2^d teams, collapse(3),                        |
|     | 4x j-loop unrolling (stride of 1   col ),                              |
|     | 4x i-loop unrolling (stride of 2^7 rows),                              |
|     | 4x k-loop unrolling,                                                   |
|     | ra: 4x data reuse,                                                     |
|     | rb: 4x data reuse,                                                     |
|     | register blocking                                                      |
|  7  | based on (2), jik-loop, 2^8 threads * 2^g teams, collapse(2)           |
|  8  | based on (7), jik-loop, 2^8 threads * 2^g teams, collapse(2),          |
|     | GPU shared memory for data re-use, 16x k-loop unrolling,               |
|     | shared memory blocking                                                 |
|  9  | based on (5), jik-loop, 2^7 threads * 2^f teams, collapse(2),          |
|     | 4x i-loop unrolling (stride of n/4 rows),                              |
|     | 4x k-loop unrolling,                                                   |
|     | rb: 4x data reuse                                                      |
| 10  | cublasSgemm in CUBLAS                                                  |

# Build

```bash
autoreconf -i; ./configure; make; make check
```

`make check` has been tested on OCuLUS (with OpenCCS) and P53s (without OpenCCS).

# Documentation

* docs/html/index.html: Source code documentation generated by Doxygen.

* docs/UserManual.md: User Manual.

