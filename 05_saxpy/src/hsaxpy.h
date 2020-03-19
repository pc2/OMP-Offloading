/**
 * @file hsaxpy.h
 * @brief Function prototype for performing the \c axpy operation on host.
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
 * @date 09.01.2020
 * @copyright CC BY-SA 2.0
 */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HSAXY_H
#define HSAXY_H

void hsaxpy(const int n,
            const float a,
            const float *x,
                  float *y);
/**<
 * @brief Performs the \c axpy operation on host.
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
 * @param n The number of elements in \p x and \p y.
 * @param a The scalar for multiplication.
 * @param x The vector \p x in \c axpy.
 * @param y The vector \p y in \c axpy.
 *
 * @return \c void.
 */

#endif

#ifdef __cplusplus
}
#endif
