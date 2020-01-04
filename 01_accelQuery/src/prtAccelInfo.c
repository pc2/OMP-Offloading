/**
 * @file prtAccelInfo.c
 * @brief Function definition for prtAccelInfo.
 *
 * This source file contains function definition for prtAccelInfo.
 *
 * @author Xin Wu (PC²)
 * @date 04.01.2020
 * @copyright GNU GPL
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <cuda_runtime.h>
#include "prtAccelInfo.h"

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
  int i;
  typedef struct {
    int SM; // 0xMm (hex notation): M = SM Major version; m = SM minor version;
    int Cores;
  } sSMtoCores;
  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60,  64},
      {0x61, 128},
      {0x62, 128},
      {0x70,  64},
      {0x72,  64},
      {0x75,  64},
      {-1, -1}};

  for (i = 0; nGpuArchCoresPerSM[i].SM != -1; i++) {
    if (nGpuArchCoresPerSM[i].SM == ((major << 4) + minor))
      return nGpuArchCoresPerSM[i].Cores;
  }
  return 999999; // Give an unreasonable value, if no value was found.
}

void prtAccelInfo(int iaccel)
{
  int corePerSM;
  struct cudaDeviceProp dev;

  cudaSetDevice(iaccel);
  cudaGetDeviceProperties(&dev, iaccel);
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
