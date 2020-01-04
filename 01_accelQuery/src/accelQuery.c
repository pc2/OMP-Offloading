/**
 * @file accelQuery.c
 *
 * @mainpage accelQuery
 *
 * @author Xin Wu (PC²)
 * @date 04.01.2020
 * @copyright GNU GPL
 *
 * accelQuery searches accelerator(s) on a heterogeneous computer.
 * Accelerator(s), if found, will be enumerated with some basic info.
 */

#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "prtAccelInfo.h"

/**
 * @brief Main entry point for accelQuery.
 */
int main(int argc, char *argv[])
{
  int iaccel, naccel;

#pragma omp target
{
  if (omp_is_initial_device()) {
    printf("Hello World from Host.\n");
  } else {
    printf("Hello World from Accelerator(s).\n");
  }
}
  // no accelerator
  if (0 == (naccel = omp_get_num_devices())) return 0;
  // one or more accelerator(s)
  printf("\n%d Accelerator(s) found\n", naccel);
  for (iaccel = 0; iaccel < naccel; iaccel++) {
    prtAccelInfo(iaccel);
  }
  return 0;
}
