/**
 * @file scalarAddition.c
 *
 * @brief scalarAddition adds two integers on host and accelerator, and also
 * compares the performance.
 *
 * Offload to GPU:
 * gcc -Wall -fopenmp -foffload=nvptx-none scalarAddition.c
 *
 */

#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

/**
 * @brief Main entry point for scalarAddition.
 */
int main(int argc, char *argv[])
{
  /*
   * data on host
   */
  int a, b, c, // c = a + b;
         y, z; // z = x + y; (x in device data environment)
  struct timespec rt[2];

  /*
   * scalar addition on host
   */
  clock_gettime(CLOCK_REALTIME, rt + 0);
  a = 2;
  b = 4;
  c = a + b;
  clock_gettime(CLOCK_REALTIME, rt + 1);
  printf("scalar addition on host: %12.9f s\n",
      (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec));
  /*
   * scalar addition on accelerator
   */
  y = 4;
  clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target map(to:y) map(from:z)
{
  int x; // only accessible from accelerator
  x = 2;
  z = x + y;
}
  clock_gettime(CLOCK_REALTIME, rt + 1);
  printf("scalar addition on accelerator: %12.9f s\n",
      (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec));
  /*
   * Question: How to measure the walltime for H-A data transfer rate? FIXME
   * Question: How to measure the walltime for a kernel launch on GPU? FIXME
   * Question: How to monitor this tiny calculation on GPU?            FIXME
   */
  /*
   * check the result
   */
  assert(c == z);
  return 0;
}
