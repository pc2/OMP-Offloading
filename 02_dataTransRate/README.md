---
title: dataTransRate
author: Xin Wu (PC²)
date: 07.01.2020
---

# Introduction

`dataTransRate` gives the data transfer rate (in MB/sec) from `src` to `dst`.

The possible situations are:

* h2h: `src` = host  and `dst` = host
* h2a: `src` = host  and `dst` = accel
* a2a: `src` = accel and `dst` = accel

# Build

```bash
autoreconf -i; ./configure; make; make check
```

`make check` has been tested on OCuLUS (with OpenCCS) and P53s (without OpenCCS).

# Documentation

* docs/html/index.html: Source code documentation generated by Doxygen.

* docs/UserManual.md: User Manual.

