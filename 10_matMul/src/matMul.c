/**
 * @file matMul.c
 *
 * @mainpage matMul
 *
 * @author Xin Wu (PC²)
 * @date 19.03.2020
 * @copyright CC BY-SA 2.0
 *
 * matMul performs matrix multiplication in single-precision on GPU. The
 * performance (in GFLOPS) for different implementations is compared and the
 * numerical results are also verified.
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
#include "matMulAB.h"

#define NLUP (16)

/**
 * @brief Main entry point for matMul.
 */
int main(int argc, char *argv[])
{
  int    ial, idx, n,
         iret = 0;
  size_t n2bytes;
  float  *a, *b, *c,
         *chost, // c matrix on host (as reference)
         *caccl, // c matrix on accl
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
  if (NULL == (c     = (float *) mkl_malloc(n2bytes, (16 * 256)))) iret = -1;
  if (NULL == (chost = (float *) mkl_malloc(n2bytes, (16 * 256)))) iret = -1;
  if (NULL == (caccl = (float *) mkl_malloc(n2bytes, (16 * 256)))) iret = -1;
  if (iret != 0) {
    printf("error: memory allocation\n");
    mkl_free(a); mkl_free(b); mkl_free(c);
    mkl_free(chost); mkl_free(caccl);
    exit(EXIT_FAILURE);
  }
#pragma omp parallel for default(none) \
  shared(a, b, c, chost, caccl, n) private(idx)
  for (idx = 0; idx < n * n; idx++) {
    a[idx] = rand() % 32 / 32.0f;
    b[idx] = rand() % 32 / 32.0f;
    c[idx] = rand() % 32 / 32.0f;
    chost[idx] = 0.0f;
    caccl[idx] = 0.0f;
  }
  printf("matrix dim: %d x %d\ntime averaged over %d loops\n", n, n, NLUP);
  /*
   * matMul on host
   */
  memcpy(chost, c, n2bytes);
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
      n, n, n, 1.0f, a, n, b, n, 1.0f, chost, n);
  /*
   * matMul on accl
   */
  for (ial = 0; ial <99; ial++) {
    wtcalc = -1.0;
    /* 
     * See matMulAB.c for details:
     *
     * ial:
     *
     * 0: jik-loop, 2^9 threads x 2^3 teams,
     *    uncoalesced memory access
     *
     * 1: jki-loop, 2^9 threads x 2^3 teams,
     *    uncoalesced memory access, uncoalesced r&w in innermost loop
     *
     * 2: jik-loop, 2^9 threads x 2^f teams, collapse(2)
     *
     * 3: jki-loop, 2^9 threads x 2^f teams, collapse(2),
     *    race condition for writing c!
     *
     * 4: jik-loop, 2^9 threads x 2^f teams, collapse(2),
     *    4x k-loop unrolling
     *
     * 5: jik-loop, 2^9 threads x 2^f teams, collapse(3),
     *    4x i-loop unrolling (2x + 2x),
     *    4x k-loop unrolling,
     *    rb: 4x data reuse
     *
     * 6: jik-loop, 2^9 threads x 2^f teams, collapse(3),
     *    2x j-loop unrolling,
     *    4x i-loop unrolling (2x + 2x),
     *    4x k-loop unrolling,
     *    ra: 2x data reuse,
     *    rb: 4x data reuse
     *
     * otherwise: cublasSgemm in CUBLAS
     */
    memcpy(caccl, c, n2bytes);
    matMulAB_accl(a, b, caccl, n, ial); // 1st run not counted
    maxabserr = -1.0f;
    for (idx = 0; idx < n * n; idx++) {
      maxabserr = fabsf(caccl[idx] - chost[idx]) > maxabserr?
                  fabsf(caccl[idx] - chost[idx]) : maxabserr;
    }
    matMulAB_accl(a, b, caccl, n, ial); // 2nd run not counted
    // timing : start
    wtcalc = 0.0;
    clock_gettime(CLOCK_REALTIME, rt + 0);
    for (int i = 0; i < NLUP; ++i) {
      matMulAB_accl(a, b, caccl, n, ial);
    }
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt=(rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("matmulAB on accl (%d) : %9.1f GFLOPS; maxabserr = %9.6f",
        ial, NLUP * 2.0e-9 * n * n * n / wt, maxabserr);
    if (wtcalc > 0.0) {
      ial = 100;
      printf("; %9.1f GFLOPS\n", NLUP * 2.0e-9 * n * n * n / wtcalc);
      printf("\nTschüß\n");
    } else {
      printf("\n");
    }
  }
  /*
   * release memory
   */
  mkl_free(a); mkl_free(b); mkl_free(c);
  mkl_free(chost); mkl_free(caccl);
  return 0;
}
