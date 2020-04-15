/**
 * @file matAddAB.c
 *
 * @brief Function definition for matrix addition (A += B) in single-precision.
 *
 * This source file contains function definition for matrix addition (A += B)
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
#include "matAddAB.h"

#define NTHRDS7 (1 << 0x7) /* 2^{7}  */
#define NTHRDS8 (1 << 0x8) /* 2^{8}  */
#define NTHRDS9 (1 << 0x9) /* 2^{9}  */

#define LTEAMSD (1 << 0xD) /* 2^{13} */
#define LTEAMSE (1 << 0xE) /* 2^{14} */
#define LTEAMSF (1 << 0xF) /* 2^{15} */

#define BLKROW  (512) /* 2x number of threads in each team */

double wtcalc;

void matAddAB_accl(float *a,
                   float *b,
                   int n,
                   int ial)
{
  cublasHandle_t handle;
  float alfa   = 1.0f,
        *a_dev = NULL,
        *b_dev = NULL;
  struct timespec rt[2];

  switch (ial) {
    case 0:
/*
 * - ij-loop
 * - 2^9 threads per team and 2^3 teams
 * - coalesced memory access
 */
#pragma omp target data  device(0) \
  map(to:n, b[0:n * n]) map(tofrom:a[0:n * n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams device(0) num_teams(LTEAMSF) thread_limit(NTHRDS9) \
  map(to:n, b[0:n * n]) map(tofrom:a[0:n * n]) \
  default(none) shared(a, b, n)
#pragma omp distribute parallel for num_threads(NTHRDS9) \
  dist_schedule(static, NTHRDS9) \
  default(none) shared(a, b, n)
for (int i = 0; i < n; ++i) { /* parallel */
for (int j = 0; j < n; ++j) { /* sequential */
  a[j * n + i] += b[j * n + i];
} /* end j-loop */
} /* end i-loop */
  clock_gettime(CLOCK_REALTIME, rt + 1);
}
      break;
    case 1:
/*
 * - ji-loop
 * - 2^9 threads per team and 2^3 teams
 * - n-stride memory read  for a and b
 * - n-stride memory write for a
 */
#pragma omp target data  device(0) \
  map(to:n, b[0:n * n]) map(tofrom:a[0:n * n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams device(0) num_teams(LTEAMSF) thread_limit(NTHRDS9) \
  map(to:n, b[0:n * n]) map(tofrom:a[0:n * n]) \
  default(none) shared(a, b, n)
#pragma omp distribute parallel for num_threads(NTHRDS9) \
  dist_schedule(static, NTHRDS9) \
  default(none) shared(a, b, n)
for (int j = 0; j < n; ++j) { /* parallel */
for (int i = 0; i < n; ++i) { /* sequential */
  a[j * n + i] += b[j * n + i];
} /* end i-loop */
} /* end j-loop */
  clock_gettime(CLOCK_REALTIME, rt + 1);
}
      break;
    case 2:
/*
 * - ij-loop
 * - 2^9 threads per team and 2^f teams
 * - collapse(2)
 * - n-stride memory read  for a and b
 * - n-stride memory write for a
 */
#pragma omp target data  device(0) \
  map(to:n, b[0:n * n]) map(tofrom:a[0:n * n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams device(0) num_teams(LTEAMSF) thread_limit(NTHRDS9) \
  map(to:n, b[0:n * n]) map(tofrom:a[0:n * n]) \
  default(none) shared(a, b, n)
#pragma omp distribute parallel for num_threads(NTHRDS9) \
  dist_schedule(static, NTHRDS9) collapse(2) \
  default(none) shared(a, b, n)
for (int i = 0; i < n; ++i) {
for (int j = 0; j < n; ++j) {
  a[j * n + i] += b[j * n + i];
} /* end j-loop */
} /* end i-loop */
  clock_gettime(CLOCK_REALTIME, rt + 1);
}
      break;
    case 3:
/*
 * - ji-loop
 * - 2^9 threads per team and 2^f teams
 * - collapse(2)
 * - coalesced memory access
 */
#pragma omp target data  device(0) \
  map(to:n, b[0:n * n]) map(tofrom:a[0:n * n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams device(0) num_teams(LTEAMSF) thread_limit(NTHRDS9) \
  map(to:n, b[0:n * n]) map(tofrom:a[0:n * n]) \
  default(none) shared(a, b, n)
#pragma omp distribute parallel for num_threads(NTHRDS9) \
  dist_schedule(static, NTHRDS9) collapse(2) \
  default(none) shared(a, b, n)
for (int j = 0; j < n; ++j) {
for (int i = 0; i < n; ++i) {
  a[j * n + i] += b[j * n + i];
} /* end i-loop */
} /* end j-loop */
  clock_gettime(CLOCK_REALTIME, rt + 1);
}
      break;
    case 4:
/*
 * - ji-loop
 * - 2^8 threads per team and 2^f teams
 * - collapse(3)
 * - 2x i-loop unrolling (stride of 2^8 rows)
 */
#pragma omp target data  device(0) \
  map(to:n, b[0:n * n]) map(tofrom:a[0:n * n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams device(0) num_teams(LTEAMSF) thread_limit(NTHRDS8) \
  map(to:n, b[0:n * n]) map(tofrom:a[0:n * n]) \
  default(none) shared(a, b, n)
#pragma omp distribute parallel for num_threads(NTHRDS8) \
  dist_schedule(static, NTHRDS8) collapse(3) \
  default(none) shared(a, b, n)
for (int j = 0; j < n; ++j) {
for (int iblk = 0; iblk < n / BLKROW; ++iblk) {
for (int i = 0; i < NTHRDS8; ++i) { /* 2x unrolling */
  a[j * n + iblk * BLKROW + i          ] +=
  b[j * n + iblk * BLKROW + i          ];
  a[j * n + iblk * BLKROW + i + NTHRDS8] +=
  b[j * n + iblk * BLKROW + i + NTHRDS8];
} /* end i-loop */
} /* end iblk-loop */
} /* end j-loop */
  clock_gettime(CLOCK_REALTIME, rt + 1);
}
      break;
    case 5:
/*
 * - ji-loop
 * - 2^8 threads per team and 2^f teams
 * - collapse(2)
 * - 2x i-loop unrolling (stride of n/2 rows)
 */
#pragma omp target data  device(0) \
  map(to:n, b[0:n * n]) map(tofrom:a[0:n * n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams device(0) num_teams(LTEAMSF) thread_limit(NTHRDS8) \
  map(to:n, b[0:n * n]) map(tofrom:a[0:n * n]) \
  default(none) shared(a, b, n)
#pragma omp distribute parallel for num_threads(NTHRDS8) \
  dist_schedule(static, NTHRDS8) collapse(2) \
  default(none) shared(a, b, n)
for (int j = 0; j < n; ++j) {
for (int i = 0; i < (n >> 1); ++i) { /* 2x unrolling */
  a[j * n + i           ] +=
  b[j * n + i           ];
  a[j * n + i + (n >> 1)] +=
  b[j * n + i + (n >> 1)];
} /* end i-loop */
} /* end j-loop */
  clock_gettime(CLOCK_REALTIME, rt + 1);
}
      break;
    case 6:
/*
 * - ji-loop
 * - 2^8 threads per team and 2^14 teams
 * - collapse(3)
 * - 2x j-loop unrolling (stride of 1   col )
 * - 2x i-loop unrolling (stride of 2^8 rows)
 */
#pragma omp target data  device(0) \
  map(to:n, b[0:n * n]) map(tofrom:a[0:n * n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams device(0) num_teams(LTEAMSE) thread_limit(NTHRDS8) \
  map(to:n, b[0:n * n]) map(tofrom:a[0:n * n]) \
  default(none) shared(a, b, n)
#pragma omp distribute parallel for num_threads(NTHRDS8) \
  dist_schedule(static, NTHRDS8) collapse(3) \
  default(none) shared(a, b, n)
for (int j = 0; j < n; j += 2) { /* 2x unrolling */
for (int iblk = 0; iblk < n / BLKROW; ++iblk) {
for (int i = 0; i < NTHRDS8; ++i) { /* 2x unrolling */
  a[ j      * n + iblk * BLKROW + i          ] +=
  b[ j      * n + iblk * BLKROW + i          ];
  a[ j      * n + iblk * BLKROW + i + NTHRDS8] +=
  b[ j      * n + iblk * BLKROW + i + NTHRDS8];
  a[(j + 1) * n + iblk * BLKROW + i          ] +=
  b[(j + 1) * n + iblk * BLKROW + i          ];
  a[(j + 1) * n + iblk * BLKROW + i + NTHRDS8] +=
  b[(j + 1) * n + iblk * BLKROW + i + NTHRDS8];
} /* end i-loop */
} /* end iblk-loop */
} /* end j-loop */
  clock_gettime(CLOCK_REALTIME, rt + 1);
}
      break;
    default:
/*
 * cublasSaxpy in CUBLAS
 */
  if (CUBLAS_STATUS_SUCCESS != cublasCreate(&handle)) {
    printf("error: initialization (CUBLAS)\n");
    cublasDestroy(handle);
    exit(EXIT_FAILURE);
  }
  if (cudaSuccess != cudaMalloc((void **) &a_dev, sizeof(*a) * n * n) ||
      cudaSuccess != cudaMalloc((void **) &b_dev, sizeof(*b) * n * n)) {
    printf("error: memory allocation (CUDA)\n");
    cudaFree(a_dev); cudaFree(b_dev);
    cublasDestroy(handle);
    exit(EXIT_FAILURE);
  }
  if (CUBLAS_STATUS_SUCCESS != cublasSetMatrix(n, n, sizeof(*a), a, n, a_dev, n) ||
      CUBLAS_STATUS_SUCCESS != cublasSetMatrix(n, n, sizeof(*b), b, n, b_dev, n)) {
    printf("error: host --> accl (CUBLAS)\n");
    cudaFree(a_dev); cudaFree(b_dev);
    cublasDestroy(handle);
    exit(EXIT_FAILURE);
  }
  clock_gettime(CLOCK_REALTIME, rt + 0);
  if (CUBLAS_STATUS_SUCCESS != cublasSaxpy(handle, n * n, &alfa, b_dev, 1, a_dev, 1)) {
    printf("error: cublasSaxpy (CUBLAS)\n");
    cudaFree(a_dev); cudaFree(b_dev);
    cublasDestroy(handle);
    exit(EXIT_FAILURE);
  }
  if (cudaSuccess != cudaDeviceSynchronize()) {
    printf("error: device synchronization (CUDA)\n");
    cudaFree(a_dev); cudaFree(b_dev);
    cublasDestroy(handle);
    exit(EXIT_FAILURE);
  }
  clock_gettime(CLOCK_REALTIME, rt + 1);
  if (CUBLAS_STATUS_SUCCESS != cublasGetMatrix(n, n, sizeof(*a), a_dev, n, a, n)) {
    printf("error: accl --> host (CUBLAS)\n");
    cudaFree(a_dev); cudaFree(b_dev);
    cublasDestroy(handle);
    exit(EXIT_FAILURE);
  }
  cudaFree(a_dev); cudaFree(b_dev);
  cublasDestroy(handle);
      break;
  } /* end switch (ial) */
  if (wtcalc >= 0.0) {
    wtcalc += (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
  }
}
