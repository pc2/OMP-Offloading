/**
 * @file saxpy.c
 *
 * @mainpage saxpy
 *
 * @author Xin Wu (PC²)
 * @date 05.04.2020
 * @copyright CC BY-SA 2.0
 *
 * saxpy performs the \c saxpy operation on host as well as accelerator.
 * The performance (in MB/s) for different implementations is also compared.
 *
 * The \c saxpy operation is defined as:
 *
 * y := a * x + y
 *
 * where:
 *
 * - a is a scalar.
 * - x and y are single-precision vectors each with n elements.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "mkl.h"
#include "hsaxpy.h"
#include "asaxpy.h"
#include "check1ns.h"
#include "wtcalc.h"

#define TWO22 (1 << 22)
#define NLUP  (32)

/**
 * @brief Main entry point for saxpy.
 */
int main(int argc, char *argv[])
{
  int    i, n,
         iret,
         ial;
  size_t nbytes;
  float  a = 2.0f,
         *x, *y,
         *yhost,
         *yaccl,
         maxabserr;
  struct timespec rt[2];
  double wt; // walltime

  /*
   * We need 1 ns time resolution.
   */
  check1ns();
  printf("The system supports 1 ns time resolution\n");
  /*
   * check the number of accelerators
   */
  if (0 == omp_get_num_devices()) {
    printf("No accelerator found ... exit\n");
    exit(EXIT_FAILURE);
  }
  /*
   * preparation
   */
  n      = TWO22;
  nbytes = sizeof(float) * n;
  iret   = 0;
  if (NULL == (x     = (float *) mkl_malloc(nbytes, (16 * 256)))) iret = -1;
  if (NULL == (y     = (float *) mkl_malloc(nbytes, (16 * 256)))) iret = -1;
  if (NULL == (yhost = (float *) mkl_malloc(nbytes, (16 * 256)))) iret = -1;
  if (NULL == (yaccl = (float *) mkl_malloc(nbytes, (16 * 256)))) iret = -1;
  if (0 != iret) {
    printf("error: memory allocation\n");
    mkl_free(x);     mkl_free(y);
    mkl_free(yhost); mkl_free(yaccl);
    exit(EXIT_FAILURE);
  }
#pragma omp parallel for default(none) \
  shared(a, x, y, yhost, yaccl, n) private(i)
  for (i = 0; i < n; ++i) {
    x[i]     = rand() % 32 / 32.0f;
    y[i]     = rand() % 32 / 32.0f;
    yhost[i] = a * x[i] + y[i]; // yhost will be used as reference value
    yaccl[i] = 0.0f;
  }
  printf("total size of x and y is %9.1f MB\n", 2.0 * nbytes / (1 << 20));
  printf("tests are averaged over %2d loops\n", NLUP);
  /*
   * saxpy on host
   */
  for (ial = 0; ial < 2; ++ial) {
    /*
     * See hsaxpy.c for details:
     *
     * ial:
     *
     * 0: naive implementation
     * otherwise: saxpy in MKL
     */
    memcpy(yaccl, y, nbytes);
    wtcalc = -1.0;
    // skip 1st run for timing
    hsaxpy(n, a, x, yaccl, ial);
    // check yaccl
    maxabserr = -1.0f;
    for (i = 0; i < n; ++i) {
      maxabserr = fabsf(yaccl[i] - yhost[i]) > maxabserr?
                  fabsf(yaccl[i] - yhost[i]) : maxabserr;
    }
    // skip 2nd run for timing
    hsaxpy(n, a, x, yaccl, ial);
    // timing : start
    wtcalc = 0.0;
    clock_gettime(CLOCK_REALTIME, rt + 0);
    for (int ilup = 0; ilup < NLUP; ++ilup) {
      hsaxpy(n, a, x, yaccl, ial);
    }
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt=(rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("saxpy on host (%d) : %9.1f MB/s %9.1f MB/s maxabserr = %9.1f\n",
        ial, NLUP * 3.0 * nbytes / ((1 << 20) * wt),
             NLUP * 3.0 * nbytes / ((1 << 20) * wtcalc), maxabserr);
  }
  /*
   * saxpy on accl
   */
  for (ial = 1; ial < 7; ++ial) {
    /*
     * See asaxpy.c for details:
     *
     * ial:
     *
     * 0: <<<             1,   1>>>, TOO SLOW! not tested
     * 1: <<<             1, 128>>>
     * 2: <<<           128,   1>>>
     * 3: <<<           128, 128>>>
     * 4: <<<n /        128, 128>>>
     * 5: <<<n / (128 * 16), 128>>>, 16x loop unrolling
     * otherwise: cublasSaxpy in CUBLAS
     */
    memcpy(yaccl, y, nbytes);
    wtcalc = -1.0;
    // skip 1st run for timing
    asaxpy(n, a, x, yaccl, ial);
    // check yaccl
    maxabserr = -1.0f;
    for (i = 0; i < n; ++i) {
      maxabserr = fabsf(yaccl[i] - yhost[i]) > maxabserr?
                  fabsf(yaccl[i] - yhost[i]) : maxabserr;
    }
    // skip 2nd run for timing
    asaxpy(n, a, x, yaccl, ial);
    // timing : start
    wtcalc = 0.0;
    clock_gettime(CLOCK_REALTIME, rt + 0);
    for (int ilup = 0; ilup < NLUP; ++ilup) {
      asaxpy(n, a, x, yaccl, ial);
    }
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt=(rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("saxpy on accl (%d) : %9.1f MB/s %9.1f MB/s maxabserr = %9.1f\n",
        ial, NLUP * 3.0 * nbytes / ((1 << 20) * wt),
             NLUP * 3.0 * nbytes / ((1 << 20) * wtcalc), maxabserr);
  }
  /*
   * release memory
   */
  mkl_free(x);     mkl_free(y);
  mkl_free(yhost); mkl_free(yaccl);
  return 0;
}
