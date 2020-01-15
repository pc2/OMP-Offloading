/**
 * @file asaxpy.h
 * @brief Function prototype for performing the \c axpy operation on
 * accelerator.
 *
 * This header file contains function prototype for the \c axpy operation,
 * which is defined as:
 *
 * y := a * x + y
 *
 * where:
 *
 * - a is a scalar.
 * - x and y are vectors each with n elements.
 *
 * @author Xin Wu (PC²)
 * @date 15.01.2020
 * @copyright GNU GPL
 */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef ASAXY_H
#define ASAXY_H

void asaxpy(const int n,
            const float a,
            const float *x,
                  float *y,
            const int ngth);
/**<
 * @brief Performs the \c axpy operation on accelerator.
 *
 * The \c axpy operation is defined as:
 *
 * y := a * x + y
 *
 * where:
 *
 * - a is a scalar.
 * - x and y are vectors each with n elements.
 *
 * @param n    The number of elements in \p x and \p y.
 * @param a    The scalar for multiplication.
 * @param x    The vector \p x in \c axpy.
 * @param y    The vector \p y in \c axpy.
 * @param ngth The number of GPU threads.
 *
 * @return \c void.
 */

#endif

#ifdef __cplusplus
}
#endif
