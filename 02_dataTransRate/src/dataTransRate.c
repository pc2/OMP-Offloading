/**
 * @file dataTransRate.c
 *
 * @mainpage dataTransRate
 *
 * @author Xin Wu (PC²)
 * @date 07.01.2020
 * @copyright GNU GPL
 *
 * dataTransRate gives the data transfer rate (in MB/sec) from src to dst.
 *
 * The possible situations are:
 *
 * - h2h: src = host  and dst = host
 * - h2a: src = host  and dst = accel
 * - a2a: src = accel and dst = accel
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
//#include <cuda_runtime.h>
#include "check1ns.h"

#define TWO27 (1 << 27)

/**
 * @brief Main entry point for dataTransRate.
 */
int main(int argc, char *argv[])
{
  // host
  int ihost,
      *hdat[2];
  // accelerator
  int iaccel, naccel,
      *adat[2];
  size_t dat512MB;
  struct timespec rt[2];
  double wt; // walltime
  int i, iret;

  /*
   * We need 1 ns time resolution.
   */
  check1ns();
  printf("The system supports 1 ns time resolution\n");
  /*
   * check the number of accelerators
   */
#pragma omp target
{
  /*
   * There is a bug in Clang/LLVM 9.0.1:
   *
   * If an accelerator has not been activated by an OpenMP directive,
   * omp_get_num_devices() always returns 0, even if there is an accelerator.
   */
}
  naccel = omp_get_num_devices();
  if (0 == naccel) {
    printf("No accelerator found ... exit\n");
    exit(EXIT_FAILURE);
  } else {
    printf("%d accelerator found ... continue\n", naccel);
  }
  /*
   * prepare host, accel, and 512 MB data
   */
  ihost    = omp_get_initial_device(); // index of the host
  iaccel   = 0;                        // index of the 1st accel
  dat512MB = sizeof(*hdat[0]) * TWO27; // 512 MB data
  iret     = 0;
  for (i = 0; i < 2; i++) {
    if (NULL == (hdat[i] = (int *) omp_target_alloc(dat512MB, ihost))) {
      printf("error: memory allocation for hdat[%d] ...", i);
      iret = 1;
    }
    if (NULL == (adat[i] = (int *) omp_target_alloc(dat512MB, iaccel))) {
      printf("error: memory allocation for adat[%d] ...", i);
      iret = 1;
    }
  }
  if (1 == iret) {
    for (i = 0; i < 2; i++) {
      omp_target_free(hdat[i], ihost);
      omp_target_free(adat[i], iaccel);
    }
    printf(" exit\n");
    exit(EXIT_FAILURE);
  }
  for (i = 0; i < TWO27; i++) {
    (hdat[0])[i] = rand();
  }
  /*
   * data transfer rate: h2h, h2a, and a2a
   */
  printf("\nData Transfer Rate\n\n");
  printf("================================\n");
  printf(" src     dst          DTR       \n");
  printf("------- ------- ----------------\n");
  /*
   * h2h
   */
  clock_gettime(CLOCK_REALTIME, rt + 0);
  iret = omp_target_memcpy(hdat[1], hdat[0], dat512MB, 0x0, 0x0, ihost, ihost);
  clock_gettime(CLOCK_REALTIME, rt + 1);
  if (0 != iret) {
    printf("error: omp_target_memcpy (h2h)\n");
    exit(EXIT_FAILURE);
  }
  wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
  printf(" host    host   %8.1f MB/sec\n", 512.0 / wt);
  /*
   * h2a
   */
  clock_gettime(CLOCK_REALTIME, rt + 0);
  iret = omp_target_memcpy(adat[0], hdat[0], dat512MB, 0x0, 0x0, iaccel, ihost);
  clock_gettime(CLOCK_REALTIME, rt + 1);
  if (0 != iret) {
    printf("error: omp_target_memcpy (h2a)\n");
    exit(EXIT_FAILURE);
  }
  wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
  printf(" host    accel  %8.1f MB/sec\n", 512.0 / wt);
  /*
   * a2a
   */
  clock_gettime(CLOCK_REALTIME, rt + 0);
  iret = omp_target_memcpy(adat[1], adat[0], dat512MB, 0x0, 0x0, iaccel, iaccel);
  /*
   * The synchronous execution on a heterogeneous computing system can be
   * enabled by one of the following approaches:
   *
   * 1. Set the environment variable `CUDA_LAUNCH_BLOCKING`
   *
   * ```bash
   * export CUDA_LAUNCH_BLOCKING=1
   * ```
   *
   * 2. Use the CUDA API function `cudaDeviceSynchronize()` in this C code.
   *
   */
//cudaDeviceSynchronize();
  clock_gettime(CLOCK_REALTIME, rt + 1);
  if (0 != iret) {
    printf("error: omp_target_memcpy (a2a)\n");
    exit(EXIT_FAILURE);
  }
  wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
  printf(" accel   accel  %8.1f MB/sec\n", 512.0 / wt);
  printf("================================\n\n");
  /*
   * release the data
   */
  for (i = 0; i < 2; i++) {
    omp_target_free(hdat[i], ihost);
    omp_target_free(adat[i], iaccel);
  }
  return 0;
}
