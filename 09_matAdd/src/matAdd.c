/**
 * @file matAdd.c
 *
 * @mainpage matAdd
 *
 * @author Xin Wu (PC²)
 * @date 19.03.2020
 * @copyright CC BY-SA 2.0
 *
 * matAdd performs matrix addition (A += B) in single-precision on GPU.
 * The performance (in GB/s) for different implementations is compared and
 * the numerical results are also verified.
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
#include "matAddAB.h"

#define NLUP (64)

/**
 * @brief Main entry point for matAdd.
 */
int main(int argc, char *argv[])
{
  int    ial, idx, n,
         iret = 0;
  size_t n2bytes;
  float  *a, *b,
         *ahost, // a matrix on host (as reference)
         *aaccl, // a matrix on accl
         maxabserr;
  struct timespec rt[2];
  double wt; // walltime

  /*
   * preparation
   */
  n         = atoi(argv[1]); // 4096 is used for test
  n2bytes   = sizeof(float) * n * n;
  if (NULL == (a     = (float *) mkl_malloc(n2bytes, (16 * 256)))) iret = -1;
  if (NULL == (b     = (float *) mkl_malloc(n2bytes, (16 * 256)))) iret = -1;
  if (NULL == (ahost = (float *) mkl_malloc(n2bytes, (16 * 256)))) iret = -1;
  if (NULL == (aaccl = (float *) mkl_malloc(n2bytes, (16 * 256)))) iret = -1;
  if (iret != 0) {
    printf("error: memory allocation\n");
    mkl_free(a); mkl_free(b);
    mkl_free(ahost); mkl_free(aaccl);
    exit(EXIT_FAILURE);
  }
#pragma omp parallel for default(none) \
  shared(a, b, ahost, aaccl, n) private(idx)
  for (idx = 0; idx < n * n; idx++) {
    a[idx] = rand() % 32 / 32.0f;
    b[idx] = rand() % 32 / 32.0f;
    ahost[idx] = 0.0f;
    aaccl[idx] = 0.0f;
  }
  printf("matrix dim: %d x %d\ntime averaged over %d loops\n", n, n, NLUP);
  /*
   * matAdd on host (ahost will be used as ref. value for checking aaccl)
   */
  memcpy(ahost, a, n2bytes);
  cblas_saxpy(n * n, 1.0f, b, 1, ahost, 1);
  /*
   * matAdd on accl
   */
  for (ial = 0; ial < 8; ++ial) {
    /*
     * See matAddAB.c for details:
     *
     * ial:
     *
     * 0: ij-loop, 2^9 threads * 2^3 teams,
     *    coalesced memory access
     *
     * 1: ji-loop, 2^9 threads * 2^3 teams,
     *    uncoalesced memory access
     *
     * 2: ij-loop, 2^9 threads * 2^f teams, collapse(2),
     *    uncoalesced memory access
     *
     * 3: ji-loop, 2^9 threads * 2^f teams, collapse(2),
     *    coalesced memory access
     *
     * 4: ji-loop, 2^8 threads * 2^f teams, collapse(3),
     *    2x i-loop unrolling
     *
     * 5: ji-loop, 2^7 threads * 2^f teams, collapse(3),
     *    4x i-loop unrolling
     *
     * 6: ji-loop, 2^7 threads * 2^e teams, collapse(3),
     *    4x i-loop unrolling, 2x j-loop unrolling
     *
     * otherwise: cublasSaxpy in CUBLAS
     */
    memcpy(aaccl, a, n2bytes);
    wtcalc = -1.0;
    // skip 1st run for timing
    matAddAB_accl(aaccl, b, n, ial);
    // check aaccl
    maxabserr = -1.0f;
    for (idx = 0; idx < n * n; idx++) {
      maxabserr = fabsf(aaccl[idx] - ahost[idx]) > maxabserr?
                  fabsf(aaccl[idx] - ahost[idx]) : maxabserr;
    }
    // skip 2nd run for timing
    matAddAB_accl(aaccl, b, n, ial);
    // timing : start
    wtcalc = 0.0;
    clock_gettime(CLOCK_REALTIME, rt + 0);
    for (int i = 0; i < NLUP; ++i) {
      matAddAB_accl(aaccl, b, n, ial);
    }
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt=(rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("matAddAB (%d) : %9.1f GB/s %9.1f GB/s maxabserr = %9.1f\n", ial,
        NLUP * 3.0 * n2bytes / ((1 << 30) * wt),
        NLUP * 3.0 * n2bytes / ((1 << 30) * wtcalc), maxabserr);
  }
  /*
   * release memory
   */
  mkl_free(a); mkl_free(b);
  mkl_free(ahost); mkl_free(aaccl);
  return 0;
}
