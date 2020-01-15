---
title: Bugs Found in GCC
author: Xin Wu (PC²)
date: 15.01.2020
---

# Asynchronous Offloading Execution

This has not been fully implemented in GCC. See `03_taskwait`.

# Limitation of Number of GPU Threads in A Team

The number of GPU threads in a team (a contention group) is limited to 8. See
`05_saxpy_v1` and `06_saxpy_v2`.
