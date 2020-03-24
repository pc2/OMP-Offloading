/**
 * @file matAddAB.h
 *
 * @brief Function prototype for matrix addition (A += B) in single-precision.
 *
 * This header file contains function prototype for matrix addition (A += B)
 * in single-precision.
 *
 * @author Xin Wu (PC²)
 * @date 07.02.2020
 * @copyright CC BY-SA 2.0
 */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MATADDAB_H
#define MATADDAB_H

void matAddAB_accl(float *a,
                   float *b,
                   int n,
                   int ial);
/**<
 * @brief Perform matrix addition (A += B) on accl.
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
