/**
 * @file prtAccelInfo.h
 * @brief Function prototype for prtAccelInfo.
 *
 * This header file contains function prototype for prtAccelInfo.
 *
 * @author Xin Wu (PC²)
 * @date 04.01.2020
 * @copyright GNU GPL
 */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef PRTACCELINFO_H
#define PRTACCELINFO_H

void prtAccelInfo(int iaccel);
/**<
 * @brief Print some basic info of an accelerator.
 *
 * Strictly speaking, \c prtAccelInfo() can only print the basic info of an
 * Nvidia CUDA device.
 *
 * @param iaccel The index of an accelerator.
 *
 * @return \c void.
 */

#endif

#ifdef __cplusplus
}
#endif
