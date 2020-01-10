/**
 * @file saxpy.c
 *
 * @mainpage saxpy
 *
 * @author Xin Wu (PC²)
 * @date 09.01.2020
 * @copyright GNU GPL
 *
 * saxpy performs the \c axpy operation on host as well as accelerator and then
 * compares the FLOPS performance.
 *
 * The \c axpy operation is defined as:
 *
 * y := a * x + y
 *
 * where:
 *
 * - a is a scalar.
 * - x and y are vectors each with n elements.
 *
 * The initial value of \c a and elements of \c x[] and \c y[] are specially
 * designed, so that the floating-point calculations on host and accelerator can
 * be compared \e exactly.
 *
 * Please note that only <em>one GPU thread</em> is used for the \c axpy
 * calculation on accelerator in this version. This can be verified by uncomment
 * the \c CFLAGS line in \c configure.ac.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "hsaxpy.h"
#include "asaxpy.h"
#include "check1ns.h"

#define TWO02 (1 <<  2)
#define TWO04 (1 <<  4)
#define TWO08 (1 <<  8)
#define TWO27 (1 << 27)

/**
 * @brief Main entry point for saxpy.
 */
int main(int argc, char *argv[])
{
  // host
  int   i, n = TWO27,
        iret = 0;
  float a = 101.0f / TWO02,
        *x, *y,
            *z;
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
   * prepare x, y, and z
   *
   * y := a * x + y (on host)
   * z := a * x + z (on accel)
   */
  if (NULL == (x = (float *) malloc(sizeof(*x) * n))) {
    printf("error: memory allocation for 'x'\n");
    iret = -1;
  }
  if (NULL == (y = (float *) malloc(sizeof(*y) * n))) {
    printf("error: memory allocation for 'y'\n");
    iret = -1;
  }
  if (NULL == (z = (float *) malloc(sizeof(*z) * n))) {
    printf("error: memory allocation for 'z'\n");
    iret = -1;
  }
  if (0 != iret) {
    free(x);
    free(y);
    free(z);
    exit(EXIT_FAILURE);
  }
  for (i = 0; i < n; i++) {
    x[i] =        rand() % TWO04 / (float) TWO02;
    y[i] = z[i] = rand() % TWO08 / (float) TWO04;
  }
  /*
   * saxpy on host
   */
  clock_gettime(CLOCK_REALTIME, rt + 0);
  hsaxpy(n, a, x, y);
  clock_gettime(CLOCK_REALTIME, rt + 1);
  wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
  printf("saxpy on host : %9.3f sec %9.1f MFLOPS\n", wt, 2.0 * n / (1.0e6 * wt));
  /*
   * saxpy on accel
   */
  clock_gettime(CLOCK_REALTIME, rt + 0);
  asaxpy(n, a, x, z);
  clock_gettime(CLOCK_REALTIME, rt + 1);
  wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
  printf("saxpy on accel: %9.3f sec %9.1f MFLOPS\n", wt, 2.0 * n / (1.0e6 * wt));
  /*
   * check whether y[] == z[] _exactly_
   */
  for (i = 0; i < n; i++) {
    iret = *(int *) (y + i) ^ *(int *) (z + i);
    assert(iret == 0);
  }
  return 0;
}
