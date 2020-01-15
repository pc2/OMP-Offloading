/**
 * @file dataTransRate.c
 *
 * @brief dataTransRate gives the data transfer rate from src to dst.
 *
 * The possible situations are:
 *
 * - h2h: src = host  and dst = host
 * - h2a: src = host  and dst = accel
 * - a2a: src = accel and dst = accel
 *
 * Offload to GPU:
 * gcc -Wall -fopenmp -foffload=nvptx-none dataTransRate.c
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define TWO27 (1 << 27)

int main(void)
{
  // host
  int ihost,  *hdat[2];
  // accelerator
  int iaccel, *adat[2];
  size_t ndat;
  struct timespec rt[2];
  double wt; // walltime
  int i, n = TWO27, iret = 0;

  /*
   * prepare data on host and accelerator
   */
  ihost  = omp_get_initial_device(); // index of the host
  iaccel = 0;                        // index of the 1st accel
  ndat   = sizeof(*hdat[0]) * n;
  for (i = 0; i < 2; i++) {
    if (NULL == (hdat[i] = (int *) omp_target_alloc(ndat, ihost))) {
      printf("error: memory allocation for hdat[%d] ...", i);
      iret = 1;
    }
    if (NULL == (adat[i] = (int *) omp_target_alloc(ndat, iaccel))) {
      printf("error: memory allocation for adat[%d] ...", i);
      iret = 1;
    }
  }
  if (1 == iret) {
    for (i = 0; i < 2; i++) {
      omp_target_free(hdat[i], ihost);
      omp_target_free(adat[i], iaccel);
    }
    exit(EXIT_FAILURE);
  }
  for (i = 0; i < n; i++) {
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
  iret = omp_target_memcpy(hdat[1], hdat[0], ndat, 0x0, 0x0, ihost, ihost);
  clock_gettime(CLOCK_REALTIME, rt + 1);
  if (0 != iret) {
    printf("error: omp_target_memcpy (h2h)\n");
    exit(EXIT_FAILURE);
  }
  wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
  printf(" host    host   %8.1f MB/sec\n", ndat / wt);
  /*
   * h2a
   */
  clock_gettime(CLOCK_REALTIME, rt + 0);
  iret = omp_target_memcpy(adat[0], hdat[0], ndat, 0x0, 0x0, iaccel, ihost);
  clock_gettime(CLOCK_REALTIME, rt + 1);
  if (0 != iret) {
    printf("error: omp_target_memcpy (h2a)\n");
    exit(EXIT_FAILURE);
  }
  wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
  printf(" host    accel  %8.1f MB/sec\n", ndat / wt);
  /*
   * a2a
   */
  clock_gettime(CLOCK_REALTIME, rt + 0);
  iret = omp_target_memcpy(adat[1], adat[0], ndat, 0x0, 0x0, iaccel, iaccel);
  /*
   * Question: How to get the correct A-A data transfer rate? FIXME
   */
  clock_gettime(CLOCK_REALTIME, rt + 1);
  if (0 != iret) {
    printf("error: omp_target_memcpy (a2a)\n");
    exit(EXIT_FAILURE);
  }
  wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
  printf(" accel   accel  %8.1f MB/sec\n", ndat / wt);
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
