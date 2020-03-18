/**
 * @file dataTransRate.c
 *
 * @mainpage dataTransRate
 *
 * @author Xin Wu (PC²)
 * @date 12.03.2020
 * @copyright CC BY-SA 2.0
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
#include "check1ns.h"

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
  int    nMB;
  size_t data;
  struct timespec rt[2];
  double wt; // walltime
  int i, iret = 0;

  /*
   * We need 1 ns time resolution.
   */
  check1ns();
  printf("The system supports 1 ns time resolution\n");
  /*
   * check the number of accelerators
   */
  naccel = omp_get_num_devices();
  if (0 == naccel) {
    printf("No accelerator found ... exit\n");
    exit(EXIT_FAILURE);
  } else {
    printf("%d accelerator found ... continue\n", naccel);
  }
  /*
   * prepare data (default to 512 MB), host, and accel
   */
  if (1 == argc) {
    nMB = 512;
  } else {
    nMB = atoi(argv[1]);
  }
  data   = nMB * (1 << 20);
  ihost  = omp_get_initial_device(); // index of the host
  iaccel = 0;                        // index of the 1st accel
  for (i = 0; i < 2; i++) {
    if (NULL == (hdat[i] = (int *) omp_target_alloc(data, ihost))) {
      printf("error: memory allocation for hdat[%d] ...", i);
      iret = -1;
    }
    if (NULL == (adat[i] = (int *) omp_target_alloc(data, iaccel))) {
      printf("error: memory allocation for adat[%d] ...", i);
      iret = -1;
    }
  }
  if (0 != iret) {
    for (i = 0; i < 2; i++) {
      omp_target_free(hdat[i], ihost);
      omp_target_free(adat[i], iaccel);
    }
    printf(" exit\n");
    exit(EXIT_FAILURE);
  }
  printf("%d MB data will be transferred", nMB);
  for (i = 0; i < data / sizeof(*hdat[0]); i++) {
    hdat[0][i] = rand();
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
  iret = omp_target_memcpy(hdat[1], hdat[0], data, 0x0, 0x0, ihost, ihost);
  clock_gettime(CLOCK_REALTIME, rt + 1);
  if (0 != iret) {
    printf("error: omp_target_memcpy (h2h)\n");
    exit(EXIT_FAILURE);
  }
  wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
  printf(" host    host   %8.1f MB/sec\n", nMB / wt);
  /*
   * h2a
   */
  clock_gettime(CLOCK_REALTIME, rt + 0);
  iret = omp_target_memcpy(adat[0], hdat[0], data, 0x0, 0x0, iaccel, ihost);
  clock_gettime(CLOCK_REALTIME, rt + 1);
  if (0 != iret) {
    printf("error: omp_target_memcpy (h2a)\n");
    exit(EXIT_FAILURE);
  }
  wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
  printf(" host    accel  %8.1f MB/sec\n", nMB / wt);
  /*
   * a2a
   *
   * - Synchronous execution has been fixed in Clang 11.
   * - Data transfer rate is somehow lower than our expectation.
   */
  clock_gettime(CLOCK_REALTIME, rt + 0);
  iret = omp_target_memcpy(adat[1], adat[0], data, 0x0, 0x0, iaccel, iaccel);
  clock_gettime(CLOCK_REALTIME, rt + 1);
  if (0 != iret) {
    printf("error: omp_target_memcpy (a2a)\n");
    exit(EXIT_FAILURE);
  }
  wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
  printf(" accel   accel  %8.1f MB/sec\n", nMB / wt);
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
