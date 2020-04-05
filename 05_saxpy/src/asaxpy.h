/**
 * @file asaxpy.h
 * @brief Function prototype for performing the \c saxpy operation on accelerator.
 *
 * This header file contains function prototype for the \c saxpy operation,
 * which is defined as:
 *
 * y := a * x + y
 *
 * where:
 *
 * - a is a scalar.
 * - x and y are single-precision vectors each with n elements.
 *
 * @author Xin Wu (PC²)
 * @date 05.04.2020
 * @copyright CC BY-SA 2.0
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
            const int ial);
/**<
 * @brief Performs the \c saxpy operation on accelerator.
 *
 * The \c saxpy operation is defined as:
 *
 * y := a * x + y
 *
 * where:
 *
 * - a is a scalar.
 * - x and y are single-precision vectors each with n elements.
 *
 * @param n   The number of elements in \p x and \p y.
 * @param a   The scalar for multiplication.
 * @param x   The vector \p x in \c saxpy.
 * @param y   The vector \p y in \c saxpy.
 * @param ial The ial-th implementation.
 *
 * @return \c void.
 */

#endif

#ifdef __cplusplus
}
#endif
