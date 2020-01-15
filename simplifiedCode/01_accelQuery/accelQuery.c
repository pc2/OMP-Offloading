/**
 * @file accelQuery.c
 *
 * @brief accelQuery searches accelerator(s) on a heterogeneous computer.
 *
 * Host-only:
 * gcc -Wall -fopenmp -foffload=disable    accelQuery.c
 *
 * Offload to GPU:
 * gcc -Wall -fopenmp -foffload=nvptx-none accelQuery.c
 *
 */

#include <stdio.h>
#include <omp.h>

int main(void)
{
#pragma omp target
{
  if (omp_is_initial_device()) {
    printf("Hello World from Host.\n");
  } else {
    printf("Hello World from Accelerator.\n");
  }
  /*
   * Question: Why this may give _wrong_ number of accelerators? FIXME
   */
  printf("%d accelerator found.\n", omp_get_num_devices());
}
  return 0;
}
