/**
 * @file scalarAddition.c
 *
 * @brief scalarAddition adds two integers on host and accelerator, and also
 * compares the performance.
 *
 * Offload to GPU:
 * gcc -Wall -fopenmp -foffload=nvptx-none accelQuery.c
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
  printf("scalar addition on host : %ld ns\n", rt[1].tv_nsec - rt[0].tv_nsec);
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
  printf("scalar addition on accel: %ld ns\n", rt[1].tv_nsec - rt[0].tv_nsec);
  /*
   * check the result
   */
  assert(c == z);
  return 0;
}
