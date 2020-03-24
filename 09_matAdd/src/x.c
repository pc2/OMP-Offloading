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
#include "mkl.h"
#include "cublas_v2.h"
#include "matAddAB.h"

#define NTHRDS7 (1 << 0x7)
#define NTHRDS8 (1 << 0x8)
#define NTHRDS9 (1 << 0x9)

#define LTEAMSD (1 << 0xD)
#define LTEAMSE (1 << 0xE)
#define LTEAMSF (1 << 0xF)

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
  int halfn = n / 2;

  switch (ial) {
    case 0:
/*
 * - ij-loop
 * - 2^9 threads per team and 2^3 teams
 */
#pragma omp target data  device(0) \
  map(to:n, b[0:n * n]) map(tofrom:a[0:n * n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams device(0) num_teams(LTEAMSF) \
  map(to:n, b[0:n * n]) map(tofrom:a[0:n * n]) \
  default(none) shared(a, b, n)
#pragma omp distribute parallel for num_threads(NTHRDS9) \
  dist_schedule(static, NTHRDS9) \
  default(none) shared(a, b, n)
for (int i = 0; i < n; ++i) { /* parallel */
for (int j = 0; j < n; ++j) { /* sequential */
  a[j * n + i] += b[j * n + i];
} /* end i-loop */
} /* end j-loop */
  clock_gettime(CLOCK_REALTIME, rt + 1);
}
      break;
    case 0:
/*
 * - ji-loop: 2^9 threads per team and 2^3 teams
 */
#pragma omp target data  device(0) \
  map(to:n, b[0:n * n]) map(tofrom:a[0:n * n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams device(0) num_teams(LTEAMSF) \
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
    case 1:
/*
 * - 2^9 threads per team and 2^f teams
 * - collapse(2)
 */
#pragma omp target data  device(0) \
  map(to:n, b[0:n * n]) map(tofrom:a[0:n * n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams device(0) num_teams(LTEAMSF) \
  map(to:n, b[0:n * n]) map(tofrom:a[0:n * n]) \
  default(none) shared(a, b, n)
#pragma omp distribute parallel for num_threads(NTHRDS9) \
  dist_schedule(static, NTHRDS9) collapse(2) \
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
 * - 2^8 threads per team and 2^f teams
 * - collapse(3)
 * - 2x i-loop unrolling
 */
#pragma omp target data  device(0) \
  map(to:n, b[0:n * n]) map(tofrom:a[0:n * n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams device(0) num_teams(LTEAMSF) \
  map(to:n, b[0:n * n]) map(tofrom:a[0:n * n]) \
  default(none) shared(a, b, n)
#pragma omp distribute parallel for num_threads(NTHRDS7) \
  dist_schedule(static, NTHRDS9) collapse(3) \
  default(none) shared(a, b, n)
for (int j = 0; j < n; ++j) {
for (int iblk = 0; iblk < n / NTHRDS8; ++iblk) {
for (int i = iblk * NTHRDS8;
         i < iblk * NTHRDS8 + NTHRDS7; ++i) {
  a[j * n + i          ] += b[j * n + i          ];
  a[j * n + i + NTHRDS7] += b[j * n + i + NTHRDS7];
} /* end i-loop */
} /* end j-loop */
  clock_gettime(CLOCK_REALTIME, rt + 1);
}
      break;
    case 3:
/*
 * - 2^7 threads per team and 2^f teams
 * - collapse(3)
 * - 2x i-loop unrolling
 */
#pragma omp target data  device(0) \
  map(to:n, b[0:n * n]) map(tofrom:a[0:n * n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams device(0) num_teams(LTEAMSF) \
  map(to:n, b[0:n * n]) map(tofrom:a[0:n * n]) \
  default(none) shared(a, b, n)
#pragma omp distribute parallel for num_threads(NTHRDS7) \
  dist_schedule(static, NTHRDS9) collapse(3) \
  default(none) shared(a, b, n)
for (int j = 0; j < n; ++j) {
for (int iblk = 0; iblk < n / NTHRDS8; ++iblk) {
for (int i = iblk * NTHRDS8;
         i < iblk * NTHRDS8 + NTHRDS7; ++i) {
  a[j * n + i          ] += b[j * n + i          ];
  a[j * n + i + NTHRDS7] += b[j * n + i + NTHRDS7];
} /* end i-loop */
} /* end j-loop */
  clock_gettime(CLOCK_REALTIME, rt + 1);
}
      break;
    case 4:
      break;
    case 5:
      break;
    case 6:
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
