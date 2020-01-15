/**
 * @file scalarAddition.c
 *
 * @mainpage scalarAddition
 *
 * @author Xin Wu (PC²)
 * @date 08.01.2020
 * @copyright GNU GPL
 *
 * scalarAddition adds two integers on host and accelerator, and also compares
 * the performance.
 *
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "check1ns.h"

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
   * check the result
   */
  assert(c == z);
  return 0;
}
