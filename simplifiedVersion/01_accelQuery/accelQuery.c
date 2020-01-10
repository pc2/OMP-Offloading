/**
 * @file accelQuery.c
 *
 * @brief accelQuery searches accelerator(s) on a heterogeneous computer.
 *
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
}
  return 0;
}
