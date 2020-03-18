/**
 * @file gpuThreads.c
 * @brief Function definition for organizing GPU threads.
 *
 * This source file contains function definition for organizing GPU threads.
 *
 * @author Xin Wu (PC²)
 * @data 12.03.2020
 * @copyright CC BY-SA 2.0
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "gpuThreads.h"

typedef struct League {
  int itd; // index  of a thread
  int ntd; // number of threads in a team
  int itm; // index  of a team
  int ltm; // number of teams in a league
} League;

static void initLeague(League *league,
                       int ncol,
                       int nrow)
/**
 * @brief Initialize a league of GPU threads.
 *
 * Every element in a league is initialized as -1.
 *
 * @param league A league of GPU threads.
 * @param ncol   Number of columns in a league.
 * @param nrow   Number of rows    in a league.
 *
 * @return \c void.
 */
{
  int icol,
      irow;

  for (icol = 0; icol < ncol; ++icol) {
    for (irow = 0; irow < nrow; ++irow) {
      league[icol * nrow + irow].itd =
      league[icol * nrow + irow].ntd =
      league[icol * nrow + irow].itm =
      league[icol * nrow + irow].ltm = -1;
    }
  }
}

void gpuThreads(int i)
{
  League *league;
  int icol,
      irow,
      ncol,
      nrow;
  int lteams,
      nthrds;
  int wblk; /* width of unrolled loop block */

  /*
   * Initialize and assign GPU threads
   */
  switch (i)
  {
    case 0:
/*
 * 1. Dim of matrix league : 3 x 5
 * 2. Dim of GPU threads   : 3 threads/team
 *                           5 teams
 * 3. All GPU threads run thru this code block.
 *    `distribute` is not needed, because there is no for-loop.
 * 4. Each GPU thread fills the corresponding element.
 */
      ncol   = 5;
      nrow   = 3;
      lteams = 5;
      nthrds = 3;
      league = (League *) malloc(sizeof(League) * ncol * nrow);
      initLeague(league, ncol, nrow);
#pragma omp target teams device(0) num_teams(lteams) \
  map(to: nrow) map(tofrom:league[0:nrow * ncol]) \
  default(none) shared(nrow, lteams, nthrds, league)
#pragma omp parallel num_threads(nthrds) \
  default(none) shared(nrow, lteams, nthrds, league)
  {
    int itd,
        itm;
    itd = omp_get_thread_num();
    itm = omp_get_team_num();
    league[itm * nrow + itd].itd = itd;
    league[itm * nrow + itd].ntd = omp_get_num_threads();
    league[itm * nrow + itd].itm = itm;
    league[itm * nrow + itd].ltm = omp_get_num_teams();
  }
      break;
    case 1:
/*
 * 1. Dim of matrix league : 3 x 5
 * 2. Dim of GPU threads   : 3 threads/team
 *                           5 teams
 * 3. Incorrect nested loop implementation.
 * 4. The number of teams equals the number of icol-loop iterations.
 * 5. Only one thread in each team will run thru the irow-loop.
 * 6. Other threads in each team are idle.
 */
      ncol   = 5;
      nrow   = 3;
      lteams = 5;
      nthrds = 3;
      league = (League *) malloc(sizeof(League) * ncol * nrow);
      initLeague(league, ncol, nrow);
#pragma omp target teams device(0) num_teams(lteams) \
  map(to: ncol, nrow) map(tofrom:league[0:nrow * ncol]) \
  default(none) shared(ncol, nrow, lteams, nthrds, league)
#pragma omp distribute parallel for num_threads(nthrds) \
  dist_schedule(static) \
  default(none) shared(ncol, nrow, lteams, nthrds, league)
  for (int icol = 0; icol < ncol; ++icol) {
    for (int irow = 0; irow < nrow; ++irow) {
      league[icol * nrow + irow].itd = omp_get_thread_num();
      league[icol * nrow + irow].ntd = omp_get_num_threads();
      league[icol * nrow + irow].itm = omp_get_team_num();
      league[icol * nrow + irow].ltm = omp_get_num_teams();
    }
  }
      break;
    case 2:
/*
 * 1. Dim of matrix league : 3 x 5
 * 2. Dim of GPU threads   : 3 threads/team
 *                           5 teams
 * 3. The previous icol- and irow-loops are linearized manually.
 * 4. The total number of GPU threads equals the number of iterations in
 *    the linearized loop.
 * 5. All GPU threads will be distributed and fill the matrix league.
 */
      ncol   = 5;
      nrow   = 3;
      lteams = 5;
      nthrds = 3;
      league = (League *) malloc(sizeof(League) * ncol * nrow);
      initLeague(league, ncol, nrow);
#pragma omp target teams device(0) num_teams(lteams) \
  map(to: ncol, nrow) map(tofrom:league[0:nrow * ncol]) \
  default(none) shared(ncol, nrow, lteams, nthrds, league)
#pragma omp distribute parallel for num_threads(nthrds) \
  dist_schedule(static) \
  default(none) shared(ncol, nrow, lteams, nthrds, league)
  for (int idx = 0; idx < nrow * ncol; ++idx) {
    league[idx].itd = omp_get_thread_num();
    league[idx].ntd = omp_get_num_threads();
    league[idx].itm = omp_get_team_num();
    league[idx].ltm = omp_get_num_teams();
  }
      break;
    case 3:
/*
 * 1. Dim of matrix league : 3 x 5
 * 2. Dim of GPU threads   : 3 threads/team
 *                           5 teams
 * 3. Not everyone wants to linearize loops manually.
 * 4. The icol- and irow-loops are collapsed.
 * 5. All GPU threads will be distributed and fill the matrix league.
 * 6. Please note that the GPU threads are organized such that the index
 *    increases continuously with respect to irow (the loop index of the
 *    innermost loop).
 */
      ncol   = 5;
      nrow   = 3;
      lteams = 5;
      nthrds = 3;
      league = (League *) malloc(sizeof(League) * ncol * nrow);
      initLeague(league, ncol, nrow);
#pragma omp target teams device(0) num_teams(lteams) \
  map(to: ncol, nrow) map(tofrom:league[0:nrow * ncol]) \
  default(none) shared(ncol, nrow, lteams, nthrds, league)
#pragma omp distribute parallel for num_threads(nthrds) \
  dist_schedule(static) collapse(2) \
  default(none) shared(ncol, nrow, lteams, nthrds, league)
  for (int icol = 0; icol < ncol; ++icol) {
    for (int irow = 0; irow < nrow; ++irow) {
      league[icol * nrow + irow].itd = omp_get_thread_num();
      league[icol * nrow + irow].ntd = omp_get_num_threads();
      league[icol * nrow + irow].itm = omp_get_team_num();
      league[icol * nrow + irow].ltm = omp_get_num_teams();
    }
  }
      break;
    case 4:
/*
 * 1. Dim of matrix league : 7 x 7
 * 2. Dim of GPU threads   : 3 threads/team
 *                           5 teams
 * 3. The size of matrix league does not match with the number of GPU threads.
 * 4. dist_schedule(kind, chunk_size)
 *    - kind: must be static
 *    - chunk_size: When no chunk_size is specified, the iterations are divided
 *      into chunks of approximately equal in size.
 * 5. Please note that in some teams *not* all GPU threads are working!
 */
      ncol   = 7;
      nrow   = 7;
      lteams = 5;
      nthrds = 3;
      league = (League *) malloc(sizeof(League) * ncol * nrow);
      initLeague(league, ncol, nrow);
#pragma omp target teams device(0) num_teams(lteams) \
  map(to: ncol, nrow) map(tofrom:league[0:nrow * ncol]) \
  default(none) shared(ncol, nrow, lteams, nthrds, league)
#pragma omp distribute parallel for num_threads(nthrds) \
  dist_schedule(static) collapse(2) \
  default(none) shared(ncol, nrow, lteams, nthrds, league)
  for (int icol = 0; icol < ncol; ++icol) {
    for (int irow = 0; irow < nrow; ++irow) {
      league[icol * nrow + irow].itd = omp_get_thread_num();
      league[icol * nrow + irow].ntd = omp_get_num_threads();
      league[icol * nrow + irow].itm = omp_get_team_num();
      league[icol * nrow + irow].ltm = omp_get_num_teams();
    }
  }
      break;
    case 5:
/*
 * 1. Dim of matrix league : 7 x 7
 * 2. Dim of GPU threads   : 3 threads/team
 *                           5 teams
 * 3. The size of matrix league does not match with the number of GPU threads.
 * 4. dist_schedule(kind, chunk_size)
 *    - kind: must be static
 *    - chunk_size: If specified, iterations are divided into chunks of size
 *      chunk_size. Chunks are then assigned to the GPU thread teams in
 *      a round-robin fashion.
 * 5. The different ways of organizing GPU threads will impact on
 *    the performance of GPU memory access.
 */
      ncol   = 7;
      nrow   = 7;
      lteams = 5;
      nthrds = 3;
      league = (League *) malloc(sizeof(League) * ncol * nrow);
      initLeague(league, ncol, nrow);
#pragma omp target teams device(0) num_teams(lteams) \
  map(to: ncol, nrow) map(tofrom:league[0:nrow * ncol]) \
  default(none) shared(ncol, nrow, lteams, nthrds, league)
#pragma omp distribute parallel for num_threads(nthrds) \
  dist_schedule(static, nthrds) collapse(2) \
  default(none) shared(ncol, nrow, lteams, nthrds, league)
  for (int icol = 0; icol < ncol; ++icol) {
    for (int irow = 0; irow < nrow; ++irow) {
      league[icol * nrow + irow].itd = omp_get_thread_num();
      league[icol * nrow + irow].ntd = omp_get_num_threads();
      league[icol * nrow + irow].itm = omp_get_team_num();
      league[icol * nrow + irow].ltm = omp_get_num_teams();
    }
  }
      break;
    case 6:
/*
 * 1. Dim of matrix league : 12 x 6
 * 2. Dim of GPU threads   : 3 threads/team
 *                           6 teams
 * 3. icol-loop: intact
 * 4. irow-loop: CPU-like 2x loop unrolling.
 * 5. It results in uncoalesced GPU memory access and reduced performance.
 * 6. +10 to each unrolled thread is used to label the 2x irow-loop unrolling.
 */
      ncol   = 6;
      nrow   =12;
      lteams = 6;
      nthrds = 3;
      league = (League *) malloc(sizeof(League) * ncol * nrow);
      initLeague(league, ncol, nrow);
#pragma omp target teams device(0) num_teams(lteams) \
  map(to: ncol, nrow) map(tofrom:league[0:nrow * ncol]) \
  default(none) shared(ncol, nrow, lteams, nthrds, league)
#pragma omp distribute parallel for num_threads(nthrds) \
  dist_schedule(static, nthrds) collapse(2) \
  default(none) shared(ncol, nrow, lteams, nthrds, league)
  for (int icol = 0; icol < ncol; ++icol) {
    for (int irow = 0; irow < nrow; irow += 2) {
      league[icol * nrow + irow    ].itd = omp_get_thread_num();
      league[icol * nrow + irow    ].ntd = omp_get_num_threads();
      league[icol * nrow + irow    ].itm = omp_get_team_num();
      league[icol * nrow + irow    ].ltm = omp_get_num_teams();
      league[icol * nrow + irow + 1].itd = omp_get_thread_num() + 10;
      league[icol * nrow + irow + 1].ntd = omp_get_num_threads();
      league[icol * nrow + irow + 1].itm = omp_get_team_num();
      league[icol * nrow + irow + 1].ltm = omp_get_num_teams();
    }
  }
      break;
    case 7:
/*
 * 1. Dim of matrix league : 12 x 6
 * 2. Dim of GPU threads   : 3 threads/team
 *                           6 teams
 * 3. icol-loop: intact
 * 4. irow-loop: 2x loop unrolling.
 * 5. Nested loop with collapse(3).
 * 6. It features coalesced GPU memory access and good performance.
 * 7. +10 to each unrolled thread is used to label the 2x irow-loop unrolling.
 */
      ncol   = 6;
      nrow   =12;
      lteams = 6;
      nthrds = 3;
      wblk   = nthrds * 2;
      league = (League *) malloc(sizeof(League) * ncol * nrow);
      initLeague(league, ncol, nrow);
#pragma omp target teams device(0) num_teams(lteams) \
  map(to: ncol, nrow, wblk) map(tofrom:league[0:nrow * ncol]) \
  default(none) shared(ncol, nrow, wblk, lteams, nthrds, league)
#pragma omp distribute parallel for num_threads(nthrds) \
  dist_schedule(static, wblk) collapse(3) \
  default(none) shared(ncol, nrow, wblk, lteams, nthrds, league)
  for (int icol = 0; icol < ncol; ++icol) {
    for (int iblk = 0; iblk < nrow / wblk; ++iblk) {
      for (int irow = iblk * wblk;
               irow < iblk * wblk + nthrds; ++irow) {
        league[icol * nrow + irow         ].itd = omp_get_thread_num();
        league[icol * nrow + irow         ].ntd = omp_get_num_threads();
        league[icol * nrow + irow         ].itm = omp_get_team_num();
        league[icol * nrow + irow         ].ltm = omp_get_num_teams();
        league[icol * nrow + irow + nthrds].itd = omp_get_thread_num() + 10;
        league[icol * nrow + irow + nthrds].ntd = omp_get_num_threads();
        league[icol * nrow + irow + nthrds].itm = omp_get_team_num();
        league[icol * nrow + irow + nthrds].ltm = omp_get_num_teams();
      }
    }
  }
      break;
    case 8:
/*
 * 1. Dim of matrix league : 12 x 6
 * 2. Dim of GPU threads   : 3 threads/team
 *                           3 teams
 * 3. icol-loop: 2x loop unrolling.
 * 4. irow-loop: 2x loop unrolling.
 * 5. Nested loop with collapse(3).
 * 6. +10 to each unrolled team   is used to label the 2x icol-loop unrolling.
 * 7. +10 to each unrolled thread is used to label the 2x irow-loop unrolling.
 */
      ncol   = 6;
      nrow   =12;
      lteams = 3;
      nthrds = 3;
      wblk   = nthrds * 2;
      league = (League *) malloc(sizeof(League) * ncol * nrow);
      initLeague(league, ncol, nrow);
#pragma omp target teams device(0) num_teams(lteams) \
  map(to: ncol, nrow, wblk) map(tofrom:league[0:nrow * ncol]) \
  default(none) shared(ncol, nrow, wblk, lteams, nthrds, league)
#pragma omp distribute parallel for num_threads(nthrds) \
  dist_schedule(static, wblk) collapse(3) \
  default(none) shared(ncol, nrow, wblk, lteams, nthrds, league)
  for (int icol = 0; icol < ncol; icol += 2) {
    for (int iblk = 0; iblk < nrow / wblk; ++iblk) {
      for (int irow = iblk * wblk;
               irow < iblk * wblk + nthrds; ++irow) {
        league[ icol      * nrow + irow         ].itd = omp_get_thread_num();
        league[ icol      * nrow + irow         ].ntd = omp_get_num_threads();
        league[ icol      * nrow + irow         ].itm = omp_get_team_num();
        league[ icol      * nrow + irow         ].ltm = omp_get_num_teams();
        league[ icol      * nrow + irow + nthrds].itd = omp_get_thread_num() + 10;
        league[ icol      * nrow + irow + nthrds].ntd = omp_get_num_threads();
        league[ icol      * nrow + irow + nthrds].itm = omp_get_team_num();
        league[ icol      * nrow + irow + nthrds].ltm = omp_get_num_teams();
        league[(icol + 1) * nrow + irow         ].itd = omp_get_thread_num();
        league[(icol + 1) * nrow + irow         ].ntd = omp_get_num_threads();
        league[(icol + 1) * nrow + irow         ].itm = omp_get_team_num() + 10;
        league[(icol + 1) * nrow + irow         ].ltm = omp_get_num_teams();
        league[(icol + 1) * nrow + irow + nthrds].itd = omp_get_thread_num() + 10;
        league[(icol + 1) * nrow + irow + nthrds].ntd = omp_get_num_threads();
        league[(icol + 1) * nrow + irow + nthrds].itm = omp_get_team_num() + 10;
        league[(icol + 1) * nrow + irow + nthrds].ltm = omp_get_num_teams();
      }
    }
  }
      break;
    default:
      printf("Tschüß!\n");
      exit(EXIT_SUCCESS);
      break;
  }
  /*
   * Show the organization of GPU threads
   */
  printf("%dth GPU threads organization:\n", i);
  printf("\n");
  printf("No. of rows    : %3d\n", nrow);
  printf("No. of cols    : %3d\n", ncol);
  printf("No. of threads : %3d\n", league[0].ntd);
  printf("No. of teams   : %3d\n", league[0].ltm);
  printf("\n");
  for (irow = 0; irow < nrow; ++irow) {
    for (icol = 0; icol < ncol; ++icol) {
      printf("(%2d,%2d):[%2d,%2d]%s", irow, icol,
          league[icol * nrow + irow].itd,
          league[icol * nrow + irow].itm,
          icol == ncol - 1 ? "\n" : "  ");
    }
  }
  printf("\n");
  /*
   * Release the memory
   */
  free(league);
}

#ifdef __cplusplus
}
#endif
