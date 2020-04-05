/**
 * @file asaxpy.c
 * @brief Function definition for performing the \c saxpy operation on accelerator.
 *
 * This source file contains function definition for the \c saxpy operation,
 * which is defined as:
 *
 * y := a * x + y
 *
 * where:
 *
 * - a is a scalar.
 * - x and y are single-precision vectors each with n elements.
 *
 * @author Xin Wu (PC²)
 * @date 05.04.2020
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
#include "wtcalc.h"
#include "asaxpy.h"

void asaxpy(const int n,
            const float a,
            const float *x,
                  float *y,
            const int ial)
{
  cublasHandle_t handle;
  float alfa = a,
        *x_dev = NULL,
        *y_dev = NULL;
  struct timespec rt[2];
  int m = (n >> 4);

  switch (ial) {
    case 0:
/*
 * - <<<1, 1>>>
 */
#pragma omp target data  device(0) \
  map(to:a, n, x[0:n]) map(tofrom:y[0:n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams device(0) num_teams(1) \
  map(to:a, n, x[0:n]) map(tofrom:y[0:n]) \
  default(none) shared(a, n, x, y)
#pragma omp distribute parallel for num_threads(1) \
  dist_schedule(static, 1) \
  default(none) shared(a, n, x, y)
for (int i = 0; i < n; ++i) {
  y[i] = a * x[i] + y[i];
}
  clock_gettime(CLOCK_REALTIME, rt + 1);
}
      break;
    case 1:
/*
 * - <<<1, 128>>>
 */
#pragma omp target data  device(0) \
  map(to:a, n, x[0:n]) map(tofrom:y[0:n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams device(0) num_teams(1) \
  map(to:a, n, x[0:n]) map(tofrom:y[0:n]) \
  default(none) shared(a, n, x, y)
#pragma omp distribute parallel for num_threads(128) \
  dist_schedule(static, 128) \
  default(none) shared(a, n, x, y)
for (int i = 0; i < n; ++i) {
  y[i] = a * x[i] + y[i];
}
  clock_gettime(CLOCK_REALTIME, rt + 1);
}
      break;
    case 2:
/*
 * - <<<128, 1>>>
 */
#pragma omp target data  device(0) \
  map(to:a, n, x[0:n]) map(tofrom:y[0:n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams device(0) num_teams(128) \
  map(to:a, n, x[0:n]) map(tofrom:y[0:n]) \
  default(none) shared(a, n, x, y)
#pragma omp distribute parallel for num_threads(1) \
  dist_schedule(static, 1) \
  default(none) shared(a, n, x, y)
for (int i = 0; i < n; ++i) {
  y[i] = a * x[i] + y[i];
}
  clock_gettime(CLOCK_REALTIME, rt + 1);
}
      break;
    case 3:
/*
 * - <<<128, 128>>>
 */
#pragma omp target data  device(0) \
  map(to:a, n, x[0:n]) map(tofrom:y[0:n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams device(0) num_teams(128) \
  map(to:a, n, x[0:n]) map(tofrom:y[0:n]) \
  default(none) shared(a, n, x, y)
#pragma omp distribute parallel for num_threads(128) \
  dist_schedule(static, 128) \
  default(none) shared(a, n, x, y)
for (int i = 0; i < n; ++i) {
  y[i] = a * x[i] + y[i];
}
  clock_gettime(CLOCK_REALTIME, rt + 1);
}
      break;
    case 4:
/*
 * - <<<n / 128, 128>>>
 */
#pragma omp target data  device(0) \
  map(to:a, n, x[0:n]) map(tofrom:y[0:n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams device(0) num_teams((1 << 15)) \
  map(to:a, n, x[0:n]) map(tofrom:y[0:n]) \
  default(none) shared(a, n, x, y)
#pragma omp distribute parallel for num_threads(128) \
  dist_schedule(static, 128) \
  default(none) shared(a, n, x, y)
for (int i = 0; i < n; ++i) {
  y[i] = a * x[i] + y[i];
}
  clock_gettime(CLOCK_REALTIME, rt + 1);
}
      break;
    case 5:
/*
 * - <<<n / (128 * 16), 128>>>
 * - 16x loop-unrolling
 */
#pragma omp target data  device(0) \
  map(to:a, m, x[0:n]) map(tofrom:y[0:n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams device(0) num_teams(2048) \
  map(to:a, m, x[0:n]) map(tofrom:y[0:n]) \
  default(none) shared(a, m, x, y)
#pragma omp distribute parallel for num_threads(128) \
  dist_schedule(static, 128) \
  default(none) shared(a, m, x, y)
for (int i = 0; i < m; ++i) {
  y[i          ] = a * x[i          ] + y[i          ];
  y[i +       m] = a * x[i +       m] + y[i +       m];
  y[i + 0x2 * m] = a * x[i + 0x2 * m] + y[i + 0x2 * m];
  y[i + 0x3 * m] = a * x[i + 0x3 * m] + y[i + 0x3 * m];
  y[i + 0x4 * m] = a * x[i + 0x4 * m] + y[i + 0x4 * m];
  y[i + 0x5 * m] = a * x[i + 0x5 * m] + y[i + 0x5 * m];
  y[i + 0x6 * m] = a * x[i + 0x6 * m] + y[i + 0x6 * m];
  y[i + 0x7 * m] = a * x[i + 0x7 * m] + y[i + 0x7 * m];
  y[i + 0x8 * m] = a * x[i + 0x8 * m] + y[i + 0x8 * m];
  y[i + 0x9 * m] = a * x[i + 0x9 * m] + y[i + 0x9 * m];
  y[i + 0xa * m] = a * x[i + 0xa * m] + y[i + 0xa * m];
  y[i + 0xb * m] = a * x[i + 0xb * m] + y[i + 0xb * m];
  y[i + 0xc * m] = a * x[i + 0xc * m] + y[i + 0xc * m];
  y[i + 0xd * m] = a * x[i + 0xd * m] + y[i + 0xd * m];
  y[i + 0xe * m] = a * x[i + 0xe * m] + y[i + 0xe * m];
  y[i + 0xf * m] = a * x[i + 0xf * m] + y[i + 0xf * m];
}
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
  if (cudaSuccess != cudaMalloc((void **) &x_dev, sizeof(*x) * n) ||
      cudaSuccess != cudaMalloc((void **) &y_dev, sizeof(*y) * n)) {
    printf("error: memory allocation (CUDA)\n");
    cudaFree(x_dev); cudaFree(y_dev);
    cublasDestroy(handle);
    exit(EXIT_FAILURE);
  }
  if (CUBLAS_STATUS_SUCCESS != cublasSetVector(n, sizeof(*x), x, 1, x_dev, 1) ||
      CUBLAS_STATUS_SUCCESS != cublasSetVector(n, sizeof(*y), y, 1, y_dev, 1)) {
    printf("error: host --> accl (CUBLAS)\n");
    cudaFree(x_dev); cudaFree(y_dev);
    cublasDestroy(handle);
    exit(EXIT_FAILURE);
  }
  clock_gettime(CLOCK_REALTIME, rt + 0);
  if (CUBLAS_STATUS_SUCCESS != cublasSaxpy(handle, n, &alfa, x_dev, 1, y_dev, 1)) {
    printf("error: cublasSaxpy (CUBLAS)\n");
    cudaFree(x_dev); cudaFree(y_dev);
    cublasDestroy(handle);
    exit(EXIT_FAILURE);
  }
  if (cudaSuccess != cudaDeviceSynchronize()) {
    printf("error: device synchronization (CUDA)\n");
    cudaFree(x_dev); cudaFree(y_dev);
    cublasDestroy(handle);
    exit(EXIT_FAILURE);
  }
  clock_gettime(CLOCK_REALTIME, rt + 1);
  if (CUBLAS_STATUS_SUCCESS != cublasGetVector(n, sizeof(*y), y_dev, 1, y, 1)) {
    printf("error: accl --> host (CUBLAS)\n");
    cudaFree(x_dev); cudaFree(y_dev);
    cublasDestroy(handle);
    exit(EXIT_FAILURE);
  }
  cudaFree(x_dev); cudaFree(y_dev);
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
