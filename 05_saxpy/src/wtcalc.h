/**
 * @file wtcalc.h
 *
 * @brief Global variable for walltime of the calculation kernel.
 *
 * @author Xin Wu (PC²)
 * @date 05.04.2020
 * @copyright CC BY-SA 2.0
 */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef WATCALC_H
#define WATCALC_H

/*
 * wtcalc: walltime for the calculation kernel
 *
 * - wtcalc  < 0.0: reset and disable the timer
 * - wtcalc == 0.0:            enable the timer
 */
extern double wtcalc;

#endif

#ifdef __cplusplus
}
#endif
