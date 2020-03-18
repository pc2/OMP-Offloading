/**
 * @file gpuThreads.h
 * @brief Function prototype for organizing GPU threads.
 *
 * This header file contains function prototype for organizing GPU threads.
 *
 * @author Xin Wu (PC²)
 * @data 12.03.2020
 * @copyright CC BY-SA 2.0
 */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef GPUTHREADS_H
#define GPUTHREADS_H

void gpuThreads(int i);
/**<
 * @brief Show the organization of GPU threads.
 *
 * The ith organization of GPU threads is shown.
 *
 * @param i The ith organization.
 *
 * @return \c void.
 */

#endif

#ifdef __cplusplus
}
#endif
