/**
 * @file matMulAB.c
 *
 * @brief Function definition for matrix multiplication in single-precision.
 *
 * This source file contains function definition for matrix multiplication
 * in single-precision.
 *
 * @author Xin Wu (PC²)
 * @date 07.02.2020
 * @copyright CC BY-SA 2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "matMulAB.h"

#define NTHRDS7 (1 << 0x7) /* 2^{7}  */
#define NTHRDS8 (1 << 0x8) /* 2^{8}  */
#define NTHRDS9 (1 << 0x9) /* 2^{9}  */

#define LTEAMSD (1 << 0xD) /* 2^{13} */
#define LTEAMSE (1 << 0xE) /* 2^{14} */
#define LTEAMSF (1 << 0xF) /* 2^{15} */
#define LTEAMSG (1 << 020) /* 2^{16} */

#define BLKROW  (512) /* 4x number of threads in each team */
#define BLKDIM  (16)

double wtcalc;

void matMulAB_accl(float *a,
                   float *b,
                   float *c,
                   int n,
                   int ial)
{
  cublasHandle_t handle;
  float alfa   = 1.0f,
        beta   = 1.0f,
        *a_dev = NULL,
        *b_dev = NULL,
        *c_dev = NULL;
  struct timespec rt[2];

  switch (ial) {
    case 0:
/*
 * - jik-loop
 * - 2^9 threads per team and 2^3 teams
 * - n-stride memory read  for c (then in rc)
 * - n-stride memory read  for b (innermost loop)
 * - n-stride memory write for c
 */
#pragma omp target data  device(0) \
  map(to:n, a[0:n * n], b[0:n * n]) map(tofrom:c[0:n * n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams device(0) num_teams(LTEAMSF) thread_limit(NTHRDS9) \
  map(to:n, a[0:n * n], b[0:n * n]) map(tofrom:c[0:n * n]) \
  default(none) shared(a, b, c, n)
#pragma omp distribute parallel for num_threads(NTHRDS9) \
  dist_schedule(static, NTHRDS9) \
  default(none) shared(a, b, c, n)
for (int j = 0; j < n; ++j) { /* parallel */
for (int i = 0; i < n; ++i) { /* sequential */
  float rc;
  rc = c[j * n + i];
  for (int k = 0; k < n; ++k) {
    rc += a[k * n + i] * b[j * n + k];
  }
  c[j * n + i] = rc;
} /* end i-loop */
} /* end j-loop */
  clock_gettime(CLOCK_REALTIME, rt + 1);
}
      break;
    case 1:
/*
 * - jki-loop
 * - 2^9 threads per team and 2^3 teams
 * - n-stride memory read  for b (then in rb)
 * - n-stride memory read  for c (innermost loop)
 * - n-stride memory write for c (innermost loop)
 */
#pragma omp target data  device(0) \
  map(to:n, a[0:n * n], b[0:n * n]) map(tofrom:c[0:n * n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams device(0) num_teams(LTEAMSF) thread_limit(NTHRDS9) \
  map(to:n, a[0:n * n], b[0:n * n]) map(tofrom:c[0:n * n]) \
  default(none) shared(a, b, c, n)
#pragma omp distribute parallel for num_threads(NTHRDS9) \
  dist_schedule(static, NTHRDS9) \
  default(none) shared(a, b, c, n)
for (int j = 0; j < n; ++j) { /* parallel */
for (int k = 0; k < n; ++k) { /* sequential */
  float rb;
  rb = b[j * n + k];
  for (int i = 0; i < n; ++i) {
    c[j * n + i] += a[k * n + i] * rb; /* uncoalesced r&w */
  }
} /* end k-loop */
} /* end j-loop */
  clock_gettime(CLOCK_REALTIME, rt + 1);
}
      break;
    case 2:
/*
 * - jik-loop
 * - 2^9 threads per team and 2^15 teams
 * - collapse(2)
 * - no race condition
 */
#pragma omp target data  device(0) \
  map(to:n, a[0:n * n], b[0:n * n]) map(tofrom:c[0:n * n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams device(0) num_teams(LTEAMSF) thread_limit(NTHRDS9) \
  map(to:n, a[0:n * n], b[0:n * n]) map(tofrom:c[0:n * n]) \
  default(none) shared(a, b, c, n)
#pragma omp distribute parallel for num_threads(NTHRDS9) \
  dist_schedule(static, NTHRDS9) collapse(2) \
  default(none) shared(a, b, c, n)
for (int j = 0; j < n; ++j) { /* parallel */
for (int i = 0; i < n; ++i) { /* parallel */
  float rc;
  rc = c[j * n + i];
  for (int k = 0; k < n; ++k) { /* sequential */
    rc += a[k * n + i] * b[j * n + k];
  }
  c[j * n + i] = rc;
} /* end i-loop */
} /* end j-loop */
  clock_gettime(CLOCK_REALTIME, rt + 1);
}
      break;
    case 3:
/*
 * - jki-loop
 * - 2^9 threads per team and 2^15 teams
 * - collapse(2)
 * - race condition for writing c: not only one thread has the index j, a total
 *   of n GPU threads has the index j. (n / 32) warps are then scheduled on GPU.
 */
#pragma omp target data  device(0) \
  map(to:n, a[0:n * n], b[0:n * n]) map(tofrom:c[0:n * n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams device(0) num_teams(LTEAMSF) thread_limit(NTHRDS9) \
  map(to:n, a[0:n * n], b[0:n * n]) map(tofrom:c[0:n * n]) \
  default(none) shared(a, b, c, n)
#pragma omp distribute parallel for num_threads(NTHRDS9) \
  dist_schedule(static, NTHRDS9) collapse(2) \
  default(none) shared(a, b, c, n)
for (int j = 0; j < n; ++j) { /* parallel */
for (int k = 0; k < n; ++k) { /* parallel */
  float rb;
  rb = b[j * n + k];
  for (int i = 0; i < n; ++i) {
    c[j * n + i] += a[k * n + i] * rb; /* race condition between diff. warps */
  }
} /* end k-loop */
} /* end j-loop */
  clock_gettime(CLOCK_REALTIME, rt + 1);
}
      break;
    case 4:
/*
 * - jik-loop
 * - 2^9 threads per team and 2^15 teams
 * - 4x k-loop unrolling
 *
 * good: more work for one thread per iteration.
 * bad : one thread must read b 4 times in k-loop.
 *       all threads in a team do the same read of b (waste of instructions).
 * tips: each thread reads the corresponding element in b and
 *       saves it in shared memory.
 */
#pragma omp target data  device(0) \
  map(to:n, a[0:n * n], b[0:n * n]) map(tofrom:c[0:n * n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams device(0) num_teams(LTEAMSF) thread_limit(NTHRDS9) \
  map(to:n, a[0:n * n], b[0:n * n]) map(tofrom:c[0:n * n]) \
  default(none) shared(a, b, c, n)
#pragma omp distribute parallel for num_threads(NTHRDS9) \
  dist_schedule(static, NTHRDS9) collapse(2) \
  default(none) shared(a, b, c, n)
for (int j = 0; j < n; ++j) {
for (int i = 0; i < n; ++i) {
  float rc;
  rc = c[j * n + i];
  for (int k = 0; k < n; k += 4) { /* 4x unrolling */
    rc += a[ k      * n + i] * b[j * n + k    ];
    rc += a[(k + 1) * n + i] * b[j * n + k + 1];
    rc += a[(k + 2) * n + i] * b[j * n + k + 2];
    rc += a[(k + 3) * n + i] * b[j * n + k + 3];
  }
  c[j * n + i] = rc;
} /* end i-loop */
} /* end j-loop */
  clock_gettime(CLOCK_REALTIME, rt + 1);
}
      break;
    case 5:
/*
 * - jik-loop
 * - 2^7 threads per team and 2^15 teams
 * - collapse(3)
 * - 4x i-loop unrolling (stride of 2^7 rows)
 * - 4x k-loop unrolling
 * - rb: 4x data re-use
 *
 * The integer calculation of matrix indices looks ugly. But considering the GPU
 * hardware architecture, e.g. many separate INT32 units, these calculations are
 * much faster than accessing GPU global memory and save the precious registers.
 *
 */
#pragma omp target data  device(0) \
  map(to:n, a[0:n * n], b[0:n * n]) map(tofrom:c[0:n * n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams device(0) num_teams(LTEAMSF) thread_limit(NTHRDS7) \
  map(to:n, a[0:n * n], b[0:n * n]) map(tofrom:c[0:n * n]) \
  default(none) shared(a, b, c, n)
#pragma omp distribute parallel for num_threads(NTHRDS7) \
  dist_schedule(static, NTHRDS7) collapse(3) \
  default(none) shared(a, b, c, n)
for (int j = 0; j < n; ++j) {
for (int iblk = 0; iblk < n / BLKROW; ++iblk) {
for (int i = 0; i < NTHRDS7; ++i) { /* 4x unrolling */
  float rc0, rc1, rc2, rc3;
  rc0 = c[j * n + iblk * BLKROW + i              ];
  rc1 = c[j * n + iblk * BLKROW + i + NTHRDS7    ];
  rc2 = c[j * n + iblk * BLKROW + i + NTHRDS7 * 2];
  rc3 = c[j * n + iblk * BLKROW + i + NTHRDS7 * 3];
  for (int k = 0; k < n; k += 4) { /* 4x unrolling */
    /* register for b: 4x k-loop */
    float rb0, rb1, rb2, rb3;
    rb0  = b[j * n + k    ];
    rb1  = b[j * n + k + 1];
    rb2  = b[j * n + k + 2];
    rb3  = b[j * n + k + 3];
    rc0 += a[ k      * n + iblk * BLKROW + i              ] * rb0;
    rc0 += a[(k + 1) * n + iblk * BLKROW + i              ] * rb1;
    rc0 += a[(k + 2) * n + iblk * BLKROW + i              ] * rb2;
    rc0 += a[(k + 3) * n + iblk * BLKROW + i              ] * rb3;
    rc1 += a[ k      * n + iblk * BLKROW + i + NTHRDS7    ] * rb0;
    rc1 += a[(k + 1) * n + iblk * BLKROW + i + NTHRDS7    ] * rb1;
    rc1 += a[(k + 2) * n + iblk * BLKROW + i + NTHRDS7    ] * rb2;
    rc1 += a[(k + 3) * n + iblk * BLKROW + i + NTHRDS7    ] * rb3;
    rc2 += a[ k      * n + iblk * BLKROW + i + NTHRDS7 * 2] * rb0;
    rc2 += a[(k + 1) * n + iblk * BLKROW + i + NTHRDS7 * 2] * rb1;
    rc2 += a[(k + 2) * n + iblk * BLKROW + i + NTHRDS7 * 2] * rb2;
    rc2 += a[(k + 3) * n + iblk * BLKROW + i + NTHRDS7 * 2] * rb3;
    rc3 += a[ k      * n + iblk * BLKROW + i + NTHRDS7 * 3] * rb0;
    rc3 += a[(k + 1) * n + iblk * BLKROW + i + NTHRDS7 * 3] * rb1;
    rc3 += a[(k + 2) * n + iblk * BLKROW + i + NTHRDS7 * 3] * rb2;
    rc3 += a[(k + 3) * n + iblk * BLKROW + i + NTHRDS7 * 3] * rb3;
  }
  c[j * n + iblk * BLKROW + i              ] = rc0;
  c[j * n + iblk * BLKROW + i + NTHRDS7    ] = rc1;
  c[j * n + iblk * BLKROW + i + NTHRDS7 * 2] = rc2;
  c[j * n + iblk * BLKROW + i + NTHRDS7 * 3] = rc3;
} /* end i-loop */
} /* end iblk-loop */
} /* end j-loop */
  clock_gettime(CLOCK_REALTIME, rt + 1);
}
      break;
    case 6:
/*
 * - jik-loop
 * - 2^7 threads per team and 2^13 teams
 * - collapse(3)
 * - 4x j-loop unrolling (stride of 1   col )
 * - 4x i-loop unrolling (stride of 2^7 rows)
 * - 4x k-loop unrolling
 * - rb: 4x data re-use
 * - ra: 4x data re-use
 * - register blocking
 */
#pragma omp target data  device(0) \
  map(to:n, a[0:n * n], b[0:n * n]) map(tofrom:c[0:n * n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams device(0) num_teams(LTEAMSD) thread_limit(NTHRDS7) \
  map(to:n, a[0:n * n], b[0:n * n]) map(tofrom:c[0:n * n]) \
  default(none) shared(a, b, c, n)
#pragma omp distribute parallel for num_threads(NTHRDS7) \
  dist_schedule(static, NTHRDS7) collapse(3) \
  default(none) shared(a, b, c, n)
for (int j = 0; j < n; j += 4) { /* 4x unrolling */
for (int iblk = 0; iblk < n / BLKROW; ++iblk) {
for (int i = 0; i < NTHRDS7; ++i) { /* 4x unrolling */
  /* register for c: 4x j-loop * 4x i-loop */
  float rc0, rc1, rc2, rc3,
        rc4, rc5, rc6, rc7,
        rc8, rc9, rca, rcb,
        rcc, rcd, rce, rcf;
  rc0 = c[ j      * n + iblk * BLKROW + i              ];
  rc1 = c[ j      * n + iblk * BLKROW + i + NTHRDS7    ];
  rc2 = c[ j      * n + iblk * BLKROW + i + NTHRDS7 * 2];
  rc3 = c[ j      * n + iblk * BLKROW + i + NTHRDS7 * 3];
  rc4 = c[(j + 1) * n + iblk * BLKROW + i              ];
  rc5 = c[(j + 1) * n + iblk * BLKROW + i + NTHRDS7    ];
  rc6 = c[(j + 1) * n + iblk * BLKROW + i + NTHRDS7 * 2];
  rc7 = c[(j + 1) * n + iblk * BLKROW + i + NTHRDS7 * 3];
  rc8 = c[(j + 2) * n + iblk * BLKROW + i              ];
  rc9 = c[(j + 2) * n + iblk * BLKROW + i + NTHRDS7    ];
  rca = c[(j + 2) * n + iblk * BLKROW + i + NTHRDS7 * 2];
  rcb = c[(j + 2) * n + iblk * BLKROW + i + NTHRDS7 * 3];
  rcc = c[(j + 3) * n + iblk * BLKROW + i              ];
  rcd = c[(j + 3) * n + iblk * BLKROW + i + NTHRDS7    ];
  rce = c[(j + 3) * n + iblk * BLKROW + i + NTHRDS7 * 2];
  rcf = c[(j + 3) * n + iblk * BLKROW + i + NTHRDS7 * 3];
  for (int k = 0; k < n; k += 4) { /* 4x unrolling */
    /* register for b: 4x j-loop * 4x k-loop */
    float rb0, rb1, rb2, rb3,
          rb4, rb5, rb6, rb7,
          rb8, rb9, rba, rbb,
          rbc, rbd, rbe, rbf;
    rb0 = b[ j      * n + k    ];
    rb1 = b[ j      * n + k + 1];
    rb2 = b[ j      * n + k + 2];
    rb3 = b[ j      * n + k + 3];
    rb4 = b[(j + 1) * n + k    ];
    rb5 = b[(j + 1) * n + k + 1];
    rb6 = b[(j + 1) * n + k + 2];
    rb7 = b[(j + 1) * n + k + 3];
    rb8 = b[(j + 2) * n + k    ];
    rb9 = b[(j + 2) * n + k + 1];
    rba = b[(j + 2) * n + k + 2];
    rbb = b[(j + 2) * n + k + 3];
    rbc = b[(j + 3) * n + k    ];
    rbd = b[(j + 3) * n + k + 1];
    rbe = b[(j + 3) * n + k + 2];
    rbf = b[(j + 3) * n + k + 3];
    /* register for a: 4x i-loop * 4x k-loop */
    float ra0, ra1, ra2, ra3,
          ra4, ra5, ra6, ra7,
          ra8, ra9, raa, rab,
          rac, rad, rae, raf;
    ra0 = a[ k      * n + iblk * BLKROW + i              ];
    ra1 = a[ k      * n + iblk * BLKROW + i + NTHRDS7    ];
    ra2 = a[ k      * n + iblk * BLKROW + i + NTHRDS7 * 2];
    ra3 = a[ k      * n + iblk * BLKROW + i + NTHRDS7 * 3];
    ra4 = a[(k + 1) * n + iblk * BLKROW + i              ];
    ra5 = a[(k + 1) * n + iblk * BLKROW + i + NTHRDS7    ];
    ra6 = a[(k + 1) * n + iblk * BLKROW + i + NTHRDS7 * 2];
    ra7 = a[(k + 1) * n + iblk * BLKROW + i + NTHRDS7 * 3];
    ra8 = a[(k + 2) * n + iblk * BLKROW + i              ];
    ra9 = a[(k + 2) * n + iblk * BLKROW + i + NTHRDS7    ];
    raa = a[(k + 2) * n + iblk * BLKROW + i + NTHRDS7 * 2];
    rab = a[(k + 2) * n + iblk * BLKROW + i + NTHRDS7 * 3];
    rac = a[(k + 3) * n + iblk * BLKROW + i              ];
    rad = a[(k + 3) * n + iblk * BLKROW + i + NTHRDS7    ];
    rae = a[(k + 3) * n + iblk * BLKROW + i + NTHRDS7 * 2];
    raf = a[(k + 3) * n + iblk * BLKROW + i + NTHRDS7 * 3];
    /*
     * register blocking
     */
    // col 1 of c:
    rc0 += ra0 * rb0;
    rc0 += ra4 * rb1;
    rc0 += ra8 * rb2;
    rc0 += rac * rb3;
    rc1 += ra1 * rb0;
    rc1 += ra5 * rb1;
    rc1 += ra9 * rb2;
    rc1 += rad * rb3;
    rc2 += ra2 * rb0;
    rc2 += ra6 * rb1;
    rc2 += raa * rb2;
    rc2 += rae * rb3;
    rc3 += ra3 * rb0;
    rc3 += ra7 * rb1;
    rc3 += rab * rb2;
    rc3 += raf * rb3;
    // col 2 of c:
    rc4 += ra0 * rb4;
    rc4 += ra4 * rb5;
    rc4 += ra8 * rb6;
    rc4 += rac * rb7;
    rc5 += ra1 * rb4;
    rc5 += ra5 * rb5;
    rc5 += ra9 * rb6;
    rc5 += rad * rb7;
    rc6 += ra2 * rb4;
    rc6 += ra6 * rb5;
    rc6 += raa * rb6;
    rc6 += rae * rb7;
    rc7 += ra3 * rb4;
    rc7 += ra7 * rb5;
    rc7 += rab * rb6;
    rc7 += raf * rb7;
    // col 3 of c:
    rc8 += ra0 * rb8;
    rc8 += ra4 * rb9;
    rc8 += ra8 * rba;
    rc8 += rac * rbb;
    rc9 += ra1 * rb8;
    rc9 += ra5 * rb9;
    rc9 += ra9 * rba;
    rc9 += rad * rbb;
    rca += ra2 * rb8;
    rca += ra6 * rb9;
    rca += raa * rba;
    rca += rae * rbb;
    rcb += ra3 * rb8;
    rcb += ra7 * rb9;
    rcb += rab * rba;
    rcb += raf * rbb;
    // col 4 of c:
    rcc += ra0 * rbc;
    rcc += ra4 * rbd;
    rcc += ra8 * rbe;
    rcc += rac * rbf;
    rcd += ra1 * rbc;
    rcd += ra5 * rbd;
    rcd += ra9 * rbe;
    rcd += rad * rbf;
    rce += ra2 * rbc;
    rce += ra6 * rbd;
    rce += raa * rbe;
    rce += rae * rbf;
    rcf += ra3 * rbc;
    rcf += ra7 * rbd;
    rcf += rab * rbe;
    rcf += raf * rbf;
  }
  c[ j      * n + iblk * BLKROW + i              ] = rc0;
  c[ j      * n + iblk * BLKROW + i + NTHRDS7    ] = rc1;
  c[ j      * n + iblk * BLKROW + i + NTHRDS7 * 2] = rc2;
  c[ j      * n + iblk * BLKROW + i + NTHRDS7 * 3] = rc3;
  c[(j + 1) * n + iblk * BLKROW + i              ] = rc4;
  c[(j + 1) * n + iblk * BLKROW + i + NTHRDS7    ] = rc5;
  c[(j + 1) * n + iblk * BLKROW + i + NTHRDS7 * 2] = rc6;
  c[(j + 1) * n + iblk * BLKROW + i + NTHRDS7 * 3] = rc7;
  c[(j + 2) * n + iblk * BLKROW + i              ] = rc8;
  c[(j + 2) * n + iblk * BLKROW + i + NTHRDS7    ] = rc9;
  c[(j + 2) * n + iblk * BLKROW + i + NTHRDS7 * 2] = rca;
  c[(j + 2) * n + iblk * BLKROW + i + NTHRDS7 * 3] = rcb;
  c[(j + 3) * n + iblk * BLKROW + i              ] = rcc;
  c[(j + 3) * n + iblk * BLKROW + i + NTHRDS7    ] = rcd;
  c[(j + 3) * n + iblk * BLKROW + i + NTHRDS7 * 2] = rce;
  c[(j + 3) * n + iblk * BLKROW + i + NTHRDS7 * 3] = rcf;
} /* end i-loop */
} /* end iblk-loop */
} /* end j-loop */
  clock_gettime(CLOCK_REALTIME, rt + 1);
}
      break;
    case 7:
/*
 * - based on case 2
 * - jik-loop
 * - 2^8 threads per team and 2^16 teams
 * - collapse(2)
 * - no race condition
 */
#pragma omp target data  device(0) \
  map(to:n, a[0:n * n], b[0:n * n]) map(tofrom:c[0:n * n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams device(0) num_teams(LTEAMSG) thread_limit(NTHRDS8) \
  map(to:n, a[0:n * n], b[0:n * n]) map(tofrom:c[0:n * n]) \
  default(none) shared(a, b, c, n)
#pragma omp distribute parallel for num_threads(NTHRDS8) \
  dist_schedule(static, NTHRDS8) collapse(2) \
  default(none) shared(a, b, c, n)
for (int j = 0; j < n; ++j) { /* parallel */
for (int i = 0; i < n; ++i) { /* parallel */
  float rc;
  rc = c[j * n + i];
  for (int k = 0; k < n; ++k) { /* sequential */
    rc += a[k * n + i] * b[j * n + k];
  }
  c[j * n + i] = rc;
} /* end i-loop */
} /* end j-loop */
  clock_gettime(CLOCK_REALTIME, rt + 1);
}
      break;
    case 8:
/*
 * - based on case 7
 * - jik-loop
 * - 2^8 threads per team and 2^16 teams
 * - collapse(2)
 * - GPU shared memory for data re-use
 * - 16x k-loop unrolling
 */
#pragma omp target data  device(0) \
  map(to:n, a[0:n * n], b[0:n * n]) map(tofrom:c[0:n * n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams device(0) num_teams(LTEAMSG) thread_limit(NTHRDS8) \
  map(to:n, a[0:n * n], b[0:n * n]) map(tofrom:c[0:n * n]) \
  default(none) shared(a, b, c, n)
{
  // GPU shared memory for each team
  /*
   * I have tested the bank conflict-free version, but it gives worse results,
   * e.g. ~ 290 GFLOPS (20 GFLOPS less than the bank conflict version).
   * I cannot explain ...
   *
  float ashm[BLKDIM][BLKDIM + 1],
        bshm[BLKDIM][BLKDIM + 1];
   */
  float ashm[BLKDIM][BLKDIM],
        bshm[BLKDIM][BLKDIM];
#pragma omp distribute dist_schedule(static, 1) collapse(2)
for (int j = 0; j < n / BLKDIM; ++j) {
for (int i = 0; i < n / BLKDIM; ++i) {
#pragma omp parallel num_threads(NTHRDS8) \
  default(none) shared(a, b, c, n, ashm, bshm, i, j)
{
  /*
   * The code here resembles CUDA.
   */
  int td = omp_get_thread_num();
  // de-linearize the thread number
  int it, // thread number along the row
      jt; // thread number along the col
  it = td % BLKDIM;
  jt = td / BLKDIM;
  int ib, // row at the beginning of block
      jb; // col at the beginning of block
  ib = i * BLKDIM;
  jb = j * BLKDIM;
  int ii, // the real row
      jj; // the real col
  ii = ib + it;
  jj = jb + jt;
  float rc = c[jj * n + ii]; // c in register
  /*
   * the k blocks
   */
  for (int k = 0; k < n / BLKDIM; ++k) {
    // read the global data to shared memory
    ashm[jt][it] = a[(k * 16 + jt) * n + ii];
    bshm[jt][it] = b[jj * n + (k * 16 + it)];
#pragma omp barrier
    // shared memory blocking and 16x k-loop unrolling
    rc += ashm[0x0][it] * bshm[jt][0x0];
    rc += ashm[0x1][it] * bshm[jt][0x1];
    rc += ashm[0x2][it] * bshm[jt][0x2];
    rc += ashm[0x3][it] * bshm[jt][0x3];
    rc += ashm[0x4][it] * bshm[jt][0x4];
    rc += ashm[0x5][it] * bshm[jt][0x5];
    rc += ashm[0x6][it] * bshm[jt][0x6];
    rc += ashm[0x7][it] * bshm[jt][0x7];
    rc += ashm[0x8][it] * bshm[jt][0x8];
    rc += ashm[0x9][it] * bshm[jt][0x9];
    rc += ashm[0xa][it] * bshm[jt][0xa];
    rc += ashm[0xb][it] * bshm[jt][0xb];
    rc += ashm[0xc][it] * bshm[jt][0xc];
    rc += ashm[0xd][it] * bshm[jt][0xd];
    rc += ashm[0xe][it] * bshm[jt][0xe];
    rc += ashm[0xf][it] * bshm[jt][0xf];
#pragma omp barrier
  } /* end k-loop */
  c[jj * n + ii] =rc;
} /* end omp parallel */
} /* end i-loop */
} /* end j-loop */
} /* end omp target teams */
  clock_gettime(CLOCK_REALTIME, rt + 1);
}
      break;
    case 9:
/*
 * - based on case 5
 * - only diffs are listed here:
 *     * collapse(2)
 *     * 4x i-loop unrolling (stride of n/4 rows)
 */
#pragma omp target data  device(0) \
  map(to:n, a[0:n * n], b[0:n * n]) map(tofrom:c[0:n * n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams device(0) num_teams(LTEAMSF) thread_limit(NTHRDS7) \
  map(to:n, a[0:n * n], b[0:n * n]) map(tofrom:c[0:n * n]) \
  default(none) shared(a, b, c, n)
#pragma omp distribute parallel for num_threads(NTHRDS7) \
  dist_schedule(static, NTHRDS7) collapse(2) \
  default(none) shared(a, b, c, n)
for (int j = 0; j < n; ++j) {
for (int i = 0; i < (n >> 2); ++i) { /* 4x unrolling */
  float rc0, rc1, rc2, rc3;
  rc0 = c[j * n + i               ];
  rc1 = c[j * n + i + (n >> 2)    ];
  rc2 = c[j * n + i + (n >> 2) * 2];
  rc3 = c[j * n + i + (n >> 2) * 3];
  for (int k = 0; k < n; k += 4) { /* 4x unrolling */
    /* register for b: 4x k-loop */
    float rb0, rb1, rb2, rb3;
    rb0  = b[j * n + k    ];
    rb1  = b[j * n + k + 1];
    rb2  = b[j * n + k + 2];
    rb3  = b[j * n + k + 3];
    rc0 += a[ k      * n + i               ] * rb0;
    rc0 += a[(k + 1) * n + i               ] * rb1;
    rc0 += a[(k + 2) * n + i               ] * rb2;
    rc0 += a[(k + 3) * n + i               ] * rb3;
    rc1 += a[ k      * n + i + (n >> 2)    ] * rb0;
    rc1 += a[(k + 1) * n + i + (n >> 2)    ] * rb1;
    rc1 += a[(k + 2) * n + i + (n >> 2)    ] * rb2;
    rc1 += a[(k + 3) * n + i + (n >> 2)    ] * rb3;
    rc2 += a[ k      * n + i + (n >> 2) * 2] * rb0;
    rc2 += a[(k + 1) * n + i + (n >> 2) * 2] * rb1;
    rc2 += a[(k + 2) * n + i + (n >> 2) * 2] * rb2;
    rc2 += a[(k + 3) * n + i + (n >> 2) * 2] * rb3;
    rc3 += a[ k      * n + i + (n >> 2) * 3] * rb0;
    rc3 += a[(k + 1) * n + i + (n >> 2) * 3] * rb1;
    rc3 += a[(k + 2) * n + i + (n >> 2) * 3] * rb2;
    rc3 += a[(k + 3) * n + i + (n >> 2) * 3] * rb3;
  }
  c[j * n + i               ] = rc0;
  c[j * n + i + (n >> 2)    ] = rc1;
  c[j * n + i + (n >> 2) * 2] = rc2;
  c[j * n + i + (n >> 2) * 3] = rc3;
} /* end i-loop */
} /* end j-loop */
  clock_gettime(CLOCK_REALTIME, rt + 1);
}
      break;
    default:
/*
 * cublasSgemm in CUBLAS
 */
  if (CUBLAS_STATUS_SUCCESS != cublasCreate(&handle)) {
    printf("error: initialization (CUBLAS)\n");
    cublasDestroy(handle);
    exit(EXIT_FAILURE);
  }
  if (cudaSuccess != cudaMalloc((void **) &a_dev, sizeof(*a) * n * n) ||
      cudaSuccess != cudaMalloc((void **) &b_dev, sizeof(*b) * n * n) ||
      cudaSuccess != cudaMalloc((void **) &c_dev, sizeof(*c) * n * n)) {
    printf("error: memory allocation (CUDA)\n");
    cudaFree(a_dev); cudaFree(b_dev); cudaFree(c_dev);
    cublasDestroy(handle);
    exit(EXIT_FAILURE);
  }
  if (CUBLAS_STATUS_SUCCESS != cublasSetMatrix(n, n, sizeof(*a), a, n, a_dev, n) ||
      CUBLAS_STATUS_SUCCESS != cublasSetMatrix(n, n, sizeof(*b), b, n, b_dev, n) ||
      CUBLAS_STATUS_SUCCESS != cublasSetMatrix(n, n, sizeof(*c), c, n, c_dev, n)) {
    printf("error: host --> accl (CUBLAS)\n");
    cudaFree(a_dev); cudaFree(b_dev); cudaFree(c_dev);
    cublasDestroy(handle);
    exit(EXIT_FAILURE);
  }
  clock_gettime(CLOCK_REALTIME, rt + 0);
  if (CUBLAS_STATUS_SUCCESS != cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n, &alfa, a_dev, n, b_dev, n, &beta, c_dev, n)) {
    printf("error: cublasSgemm (CUBLAS)\n");
    cudaFree(a_dev); cudaFree(b_dev); cudaFree(c_dev);
    cublasDestroy(handle);
    exit(EXIT_FAILURE);
  }
  if (cudaSuccess != cudaDeviceSynchronize()) {
    printf("error: device synchronization (CUDA)\n");
    cudaFree(a_dev); cudaFree(b_dev); cudaFree(c_dev);
    cublasDestroy(handle);
    exit(EXIT_FAILURE);
  }
  clock_gettime(CLOCK_REALTIME, rt + 1);
  if (CUBLAS_STATUS_SUCCESS != cublasGetMatrix(n, n, sizeof(*c), c_dev, n, c, n)) {
    printf("error: accl --> host (CUBLAS)\n");
    cudaFree(a_dev); cudaFree(b_dev); cudaFree(c_dev);
    cublasDestroy(handle);
    exit(EXIT_FAILURE);
  }
  cudaFree(a_dev); cudaFree(b_dev); cudaFree(c_dev);
  cublasDestroy(handle);
      break;
  } /* end switch (ial) */
  if (wtcalc >= 0.0) {
    wtcalc += (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
  }
}
