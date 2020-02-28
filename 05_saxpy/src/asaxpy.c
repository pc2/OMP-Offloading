/**
 * @file asaxpy.c
 * @brief Function definition for performing the \c axpy operation on
 * accelerator.
 *
 * This source file contains function definition for the \c axpy operation,
 * which is defined as:
 *
 * y := a * x + y
 *
 * where:
 *
 * - a is a scalar.
 * - x and y are vectors each with n elements.
 *
 * @author Xin Wu (PC²)
 * @date 09.01.2020
 * @copyright GNU GPL
 */

#ifdef __cplusplus
extern "C" {
#endif

#ifdef DEBUG
#include <stdio.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#include "asaxpy.h"

void asaxpy(const int n,
            const float a,
            const float *x,
                  float *y)
{
  int i = 0;
#ifdef DEBUG
  int nth;
#endif

#ifdef DEBUG
#pragma omp target map(n, a, x[0:n], y[0:n], nth) private(i)
#else
#pragma omp target map(n, a, x[0:n], y[0:n]     ) private(i)
#endif
{
#ifdef DEBUG
  nth = omp_get_num_threads();
#endif
  for (i = 0; i < n; i++) {
    y[i] = a * x[i] + y[i];
  }
}
#ifdef DEBUG
printf("DEBUG: number of threads on accelerator: %d\n", nth);
#endif
}

#ifdef __cplusplus
}
#endif
