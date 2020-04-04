/**
 * @file prtAccelInfo.c
 * @brief Function definition for prtAccelInfo.
 *
 * This source file contains function definition for prtAccelInfo.
 *
 * @author Xin Wu (PC²)
 * @date 04.01.2020
 * @copyright CC BY-SA 2.0
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "prtAccelInfo.h"

#define CUDAErrorCheck(funcall)                                         \
do {                                                                    \
  cudaError_t ierr = funcall;                                           \
  if (cudaSuccess != ierr) {                                            \
    fprintf(stderr, "%s(line %d) : CUDA RT API error : %s(%d) -> %s\n", \
    __FILE__, __LINE__, #funcall, ierr, cudaGetErrorString(ierr));      \
    exit(ierr);                                                         \
  }                                                                     \
} while (0)

static inline int _corePerSM(int major, int minor)
/**
 * @brief Give the number of CUDA cores per streaming multiprocessor (SM).
 *
 * The number of CUDA cores per SM is determined by the compute capability.
 *
 * @param major Major revision number of the compute capability.
 * @param minor Minor revision number of the compute capability.
 *
 * @return The number of CUDA cores per SM.
 */
{
  if (1 == major) {
    if (0 == minor || 1 == minor || 2 == minor || 3 == minor) return 8;
  }
  if (2 == major) {
    if (0 == minor) return 32;
    if (1 == minor) return 48;
  }
  if (3 == major) {
    if (0 == minor || 5 == minor || 7 == minor) return 192;
  }
  if (5 == major) {
    if (0 == minor || 2 == minor) return 128;
  }
  if (6 == major) {
    if (0 == minor) return 64;
    if (1 == minor || 2 == minor) return 128;
  }
  if (7 == major) {
    if (0 == minor || 2 == minor || 5 == minor) return 64;
  }
  return -1;
}

void prtAccelInfo(int iaccel)
{
  int corePerSM;
  struct cudaDeviceProp dev;

  CUDAErrorCheck(cudaSetDevice(iaccel));
  CUDAErrorCheck(cudaGetDeviceProperties(&dev, iaccel));
  corePerSM = _corePerSM(dev.major, dev.minor);
  printf("\n");
  printf("============================================================\n");
  printf("CUDA Device name : \"%s\"\n", dev.name);
  printf("------------------------------------------------------------\n");
  printf("Comp. Capability : %d.%d\n", dev.major, dev.minor);
  printf("max clock rate   : %.0f MHz\n", dev.clockRate * 1.e-3f);
  printf("number of SMs    : %d\n", dev.multiProcessorCount);
  printf("cores  /  SM     : %d\n", corePerSM);
  printf("# of CUDA cores  : %d\n", corePerSM * dev.multiProcessorCount);
  printf("------------------------------------------------------------\n");
  printf("global memory    : %5.0f MBytes\n", dev.totalGlobalMem / 1048576.0f);
  printf("shared mem. / SM : %5.1f KBytes\n", dev.sharedMemPerMultiprocessor / 1024.0f);
  printf("32-bit reg. / SM : %d\n", dev.regsPerMultiprocessor);
  printf("------------------------------------------------------------\n");
  printf("max # of threads / SM    : %d\n", dev.maxThreadsPerMultiProcessor);
  printf("max # of threads / block : %d\n", dev.maxThreadsPerBlock);
  printf("max dim. of block        : (%d, %d, %d)\n",
      dev.maxThreadsDim[0], dev.maxThreadsDim[1], dev.maxThreadsDim[2]);
  printf("max dim. of grid         : (%d, %d, %d)\n",
      dev.maxGridSize[0],   dev.maxGridSize[1],   dev.maxGridSize[2]);
  printf("warp size                : %d\n", dev.warpSize);
  printf("============================================================\n");
}

#ifdef __cplusplus
}
#endif
