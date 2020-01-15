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
 * @date 15.01.2020
 * @copyright GNU GPL
 */

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _OPENMP
#include <omp.h>
#endif
#include "asaxpy.h"

void asaxpy(const int n,
            const float a,
            const float *x,
                  float *y,
            const int ngth)
{
  int i = 0;

#pragma omp target map(n, a, x[0:n], y[0:n]) private(i) firstprivate(ngth)
{
  #pragma omp parallel default(none) \
  shared(n, a, x, y) private(i) num_threads(ngth)
  {
  #pragma omp for
  for (i = 0; i < n; i++) {
    y[i] = a * x[i] + y[i];
  }
  }
}
}

#ifdef __cplusplus
}
#endif
