/**
 * @file matMulAB.h
 *
 * @brief Function prototype for matrix multiplication in single-precision.
 *
 * This header file contains function prototype for matrix multiplication
 * in single-precision.
 *
 * @author Xin Wu (PC²)
 * @date 07.02.2020
 * @copyright CC BY-SA 2.0
 */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MATMULAB_H
#define MATMULAB_H

void matMulAB_accl(float *a,
                   float *b,
                   float *c,
                   int n,
                   int ial);
/**<
 * @brief Perform matrix multiplication on accl.
 *
 * @return \c void.
 */

/*
 * wtcalc: walltime for the calculation kernel on GPU
 *
 * - wtcalc  < 0.0: reset and disable the timer
 * - wtcalc == 0.0:            enable the timer
 */
extern double wtcalc;

#endif

#ifdef __cplusplus
}
#endif
