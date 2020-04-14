/**
 * @file check1ns.c
 * @brief Function definition for checking 1 ns time resolution on the system.
 *
 * This source file contains function definition for checking 1 ns time
 * resolution on the system.
 *
 * @author Xin Wu (PC²)
 * @date 07.01.2020
 * @copyright CC BY-SA 2.0
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "check1ns.h"

void check1ns(void)
{
  struct timespec res;

  if (0 != clock_getres(CLOCK_REALTIME, &res)) {
    printf("error: clock_getres\n");
    exit(EXIT_FAILURE);
  }
  assert(1l == res.tv_nsec);
}
