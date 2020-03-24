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

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "matMulAB.h"

#define NTHRDS7 (1 << 0x7)
#define NTHRDS8 (1 << 0x8)
#define NTHRDS9 (1 << 0x9)

#define LTEAMSD (1 << 0xD)
#define LTEAMSE (1 << 0xE)
#define LTEAMSF (1 << 0xF)

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
  int halfn = n / 2;

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
#pragma omp target teams device(0) num_teams(LTEAMSF) \
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
#pragma omp target teams device(0) num_teams(LTEAMSF) \
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
#pragma omp target teams device(0) num_teams(LTEAMSF) \
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
#pragma omp target teams device(0) num_teams(LTEAMSF) \
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
 */
#pragma omp target data  device(0) \
  map(to:n, a[0:n * n], b[0:n * n]) map(tofrom:c[0:n * n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams device(0) num_teams(LTEAMSF) \
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
    float rb0, rb1, rb2, rb3;
    rb0 = b[j * n + k    ];
    rb1 = b[j * n + k + 1];
    rb2 = b[j * n + k + 2];
    rb3 = b[j * n + k + 3];
    rc += a[ k      * n + i] * rb0
        + a[(k + 1) * n + i] * rb1
        + a[(k + 2) * n + i] * rb2
        + a[(k + 3) * n + i] * rb3;
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
 * - 4x i-loop unrolling
 *      * 2x by number of threads
 *      * 2x by half of rows
 * - 4x k-loop unrolling
 * - rb: 4x data reuse
 */
#pragma omp target data  device(0) \
  map(to:n, halfn, a[0:n * n], b[0:n * n]) map(tofrom:c[0:n * n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams device(0) num_teams(LTEAMSF) \
  map(to:n, halfn, a[0:n * n], b[0:n * n]) map(tofrom:c[0:n * n]) \
  default(none) shared(a, b, c, n, halfn)
#pragma omp distribute parallel for num_threads(NTHRDS7) \
  dist_schedule(static, NTHRDS7) collapse(3) \
  default(none) shared(a, b, c, n, halfn)
for (int j = 0; j < n; ++j) {
for (int iblk = 0; iblk < n / NTHRDS9; ++iblk) { /* 4x unrolling */
for (int i = iblk * NTHRDS8;
         i < iblk * NTHRDS8 + NTHRDS7; ++i) {
  float rc0, rc1, rc2, rc3;
  rc0 = c[j * n + i                  ];
  rc1 = c[j * n + i + NTHRDS7        ];
  rc2 = c[j * n + i           + halfn];
  rc3 = c[j * n + i + NTHRDS7 + halfn];
  for (int k = 0; k < n; k += 4) { /* 4x unrolling */
    float rb0, rb1, rb2, rb3;
    rb0 = b[j * n + k    ];
    rb1 = b[j * n + k + 1];
    rb2 = b[j * n + k + 2];
    rb3 = b[j * n + k + 3];
    rc0+= a[ k      * n + i                  ] * rb0
        + a[(k + 1) * n + i                  ] * rb1
        + a[(k + 2) * n + i                  ] * rb2
        + a[(k + 3) * n + i                  ] * rb3;
    rc1+= a[ k      * n + i + NTHRDS7        ] * rb0
        + a[(k + 1) * n + i + NTHRDS7        ] * rb1
        + a[(k + 2) * n + i + NTHRDS7        ] * rb2
        + a[(k + 3) * n + i + NTHRDS7        ] * rb3;
    rc2+= a[ k      * n + i           + halfn] * rb0
        + a[(k + 1) * n + i           + halfn] * rb1
        + a[(k + 2) * n + i           + halfn] * rb2
        + a[(k + 3) * n + i           + halfn] * rb3;
    rc3+= a[ k      * n + i + NTHRDS7 + halfn] * rb0
        + a[(k + 1) * n + i + NTHRDS7 + halfn] * rb1
        + a[(k + 2) * n + i + NTHRDS7 + halfn] * rb2
        + a[(k + 3) * n + i + NTHRDS7 + halfn] * rb3;
  }
  c[j * n + i                  ] = rc0;
  c[j * n + i + NTHRDS7        ] = rc1;
  c[j * n + i           + halfn] = rc2;
  c[j * n + i + NTHRDS7 + halfn] = rc3;
} /* end i-loop */
} /* end iblk-loop */
} /* end j-loop */
  clock_gettime(CLOCK_REALTIME, rt + 1);
}
      break;
    case 6:
/*
 * - jik-loop
 * - 2^7 threads per team and 2^14 teams
 * - collapse(3)
 * - 2x j-loop unrolling by half of cols
 * - 4x i-loop unrolling
 *      * 2x by number of threads
 *      * 2x by half of rows
 * - 4x k-loop unrolling
 */
#pragma omp target data  device(0) \
  map(to:n, halfn, a[0:n * n], b[0:n * n]) map(tofrom:c[0:n * n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams device(0) num_teams(LTEAMSE) \
  map(to:n, halfn, a[0:n * n], b[0:n * n]) map(tofrom:c[0:n * n]) \
  default(none) shared(a, b, c, n, halfn)
#pragma omp distribute parallel for num_threads(NTHRDS7) \
  dist_schedule(static, NTHRDS7) collapse(3) \
  default(none) shared(a, b, c, n, halfn)
for (int j = 0; j < halfn; ++j) { /* 2x unrolling */
for (int iblk = 0; iblk < n / NTHRDS9; ++iblk) { /* 4x unrolling */
for (int i = iblk * NTHRDS8;
         i < iblk * NTHRDS8 + NTHRDS7; ++i) {
  /* register for c: 2x j-loop * 4x i-loop */
  float rc0, rc1, rc2, rc3,
        rc4, rc5, rc6, rc7;
  rc0 = c[ j          * n + i                  ];
  rc1 = c[ j          * n + i + NTHRDS7        ];
  rc2 = c[ j          * n + i           + halfn];
  rc3 = c[ j          * n + i + NTHRDS7 + halfn];
  rc4 = c[(j + halfn) * n + i                  ];
  rc5 = c[(j + halfn) * n + i + NTHRDS7        ];
  rc6 = c[(j + halfn) * n + i           + halfn];
  rc7 = c[(j + halfn) * n + i + NTHRDS7 + halfn];
  for (int k = 0; k < n; k += 4) { /* 4x unrolling */
    /* register for b: 2x j-loop * 4x k-loop */
    float rb0, rb1, rb2, rb3,
          rb4, rb5, rb6, rb7;
    /* register for a: 4x i-loop * 4x k-loop */
    float ra0, ra1, ra2, ra3,
          ra4, ra5, ra6, ra7,
          ra8, ra9, raa, rab,
          rac, rad, rae, raf;
    rb0 = b[ j          * n + k    ];
    rb1 = b[ j          * n + k + 1];
    rb2 = b[ j          * n + k + 2];
    rb3 = b[ j          * n + k + 3];
    rb4 = b[(j + halfn) * n + k    ];
    rb5 = b[(j + halfn) * n + k + 1];
    rb6 = b[(j + halfn) * n + k + 2];
    rb7 = b[(j + halfn) * n + k + 3];
    ra0 = a[ k      * n + i                  ];
    ra1 = a[(k + 1) * n + i                  ];
    ra2 = a[(k + 2) * n + i                  ];
    ra3 = a[(k + 3) * n + i                  ];
    ra4 = a[ k      * n + i + NTHRDS7        ];
    ra5 = a[(k + 1) * n + i + NTHRDS7        ];
    ra6 = a[(k + 2) * n + i + NTHRDS7        ];
    ra7 = a[(k + 3) * n + i + NTHRDS7        ];
    ra8 = a[ k      * n + i           + halfn];
    ra9 = a[(k + 1) * n + i           + halfn];
    raa = a[(k + 2) * n + i           + halfn];
    rab = a[(k + 3) * n + i           + halfn];
    rac = a[ k      * n + i + NTHRDS7 + halfn];
    rad = a[(k + 1) * n + i + NTHRDS7 + halfn];
    rae = a[(k + 2) * n + i + NTHRDS7 + halfn];
    raf = a[(k + 3) * n + i + NTHRDS7 + halfn];
    /*
     * register blocking
     */
    rc0+= ra0 * rb0
        + ra1 * rb1
        + ra2 * rb2
        + ra3 * rb3;
    rc1+= ra4 * rb0
        + ra5 * rb1
        + ra6 * rb2
        + ra7 * rb3;
    rc2+= ra8 * rb0
        + ra9 * rb1
        + raa * rb2
        + rab * rb3;
    rc3+= rac * rb0
        + rad * rb1
        + rae * rb2
        + raf * rb3;
    rc4+= ra0 * rb4
        + ra1 * rb5
        + ra2 * rb6
        + ra3 * rb7;
    rc5+= ra4 * rb4
        + ra5 * rb5
        + ra6 * rb6
        + ra7 * rb7;
    rc6+= ra8 * rb4
        + ra9 * rb5
        + raa * rb6
        + rab * rb7;
    rc7+= rac * rb4
        + rad * rb5
        + rae * rb6
        + raf * rb7;
  }
  c[ j          * n + i                  ] = rc0;
  c[ j          * n + i + NTHRDS7        ] = rc1;
  c[ j          * n + i           + halfn] = rc2;
  c[ j          * n + i + NTHRDS7 + halfn] = rc3;
  c[(j + halfn) * n + i                  ] = rc4;
  c[(j + halfn) * n + i + NTHRDS7        ] = rc5;
  c[(j + halfn) * n + i           + halfn] = rc6;
  c[(j + halfn) * n + i + NTHRDS7 + halfn] = rc7;
} /* end i-loop */
} /* end iblk-loop */
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

#ifdef __cplusplus
}
#endif
