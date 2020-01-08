/**
 * @file taskwait.c
 *
 * @mainpage taskwait
 *
 * @author Xin Wu (PC²)
 * @date 08.01.2020
 * @copyright GNU GPL
 *
 * taskwait checks the taskwait construct for the deferred target task. At the
 * time of writing, this hasn't been implemented in the GCC 9.2 compiler.
 */

#include <assert.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Main entry point for taskwait.
 */
int main(int argc, char *argv[])
{
  int a, b, c,
      x, y, z;

  a = x = 2;
  b = y = 4;
#pragma omp target map(a, b, c) nowait
{
  c = a + b; /* This is executed on accelerator. */
}
  z = x + y; /* This is executed on host.        */
#pragma omp taskwait
  assert(c == z);
  return 0;
}
