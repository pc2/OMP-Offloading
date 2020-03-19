/**
 * @file check1ns.h
 * @brief Function prototype for checking 1 ns time resolution on the system.
 *
 * This header file contains function prototype for checking 1 ns time
 * resolution on the system.
 *
 * @author Xin Wu (PC²)
 * @date 07.01.2020
 * @copyright CC BY-SA 2.0
 */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef CHECK1NS_H
#define CHECK1NS_H

void check1ns(void);
/**<
 * @brief Check whether 1 ns time resolution is available on the system.
 *
 * We need 1 ns time resolution. If it's available, program continues normally.
 * Otherwise, program terminates.
 *
 * @return \c void.
 */

#endif

#ifdef __cplusplus
}
#endif
