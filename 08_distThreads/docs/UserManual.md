---
title: distThreads
author: Xin Wu (PC²)
date: 12.03.2020
---

# Introduction

`distThreads` demonstrates the organization of threads and teams in a league on
GPU.

* Column-major is assumed thru the entire code!

* The following tables only summarize the most important points. For more
  details on the ith organization of the GPU threads see comments in
  `gpuThreads.c`.

| i |  matrix league  |     GPU threads     |
|:-:|:---------------:|:-------------------:|
|   |  nrow  x  ncol  |  nthrds  x  lteams  |
| 0 |    3   x    5   |    3     x    5     |
| 1 |    3   x    5   |    3     x    5     |
| 2 |    3   x    5   |    3     x    5     |
| 3 |    3   x    5   |    3     x    5     |
| 4 |    7   x    7   |    3     x    5     |
| 5 |    7   x    7   |    3     x    5     |
| 6 |   12   x    6   |    3     x    6     |
| 7 |   12   x    6   |    3     x    6     |
| 8 |   12   x    6   |    3     x    3     |

| i |  Remarks                                                        |
|:-:|:----------------------------------------------------------------|
| 0 | Used as Reference. No loop at all.                              |
| 1 | Incorrect nested loop impl.                                     |
| 2 | Correct impl. Manually linearized loop.                         |
| 3 | Correct impl. Nested loop with collapse(2).                     |
| 4 | Irreg. matrix. Default chunk_size. Some GPU threads are idle.   |
| 5 | Irreg. matrix. chunk_size = nthrds. Better performance.         |
| 6 | CPU-like 2x irow-loop unrolling. Uncoalesced GPU memory access. |
| 7 | 2x irow-loop unrolling. Nested loop with collapse(3).           |
|   | Coalesced GPU memory access.                                    |
| 8 | 2x icol-loop unrolling. 2x irow-loop unrolling.                 |
|   | Nested loop with collapse(3). Best Performance.                 |

# Usage

```bash
distThreads
```

