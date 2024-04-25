//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include "generate_matrix.hpp"
#ifdef USE_MKL
#include <mkl.h>
#endif

// We map to KokkosKernels to get access to third party libraries
#ifdef USE_KOKKOS_KERNELS
#include <KokkosBlas.hpp>
#include <KokkosSparse_spmv.hpp>

template <class YType, class AType, class XType>
void spmv(YType y, AType A, XType x) {
  KokkosSparse::CrsMatrix<
      double, INT_TYPE,
      Kokkos::Device<Kokkos::DefaultExecutionSpace,
                     typename Kokkos::DefaultExecutionSpace::memory_space>,
      void, INT_TYPE>
      matrix("A",          // const std::string& /* label */,
             A.num_rows(), // const OrdinalType nrows,
             A.num_cols(), // const OrdinalType ncols,
             A.nnz(),      // const size_type annz,
             A.values,     // const values_type& vals,
             A.row_ptr,    // const row_map_type& rowmap,
             A.col_idx);   // const index_type& cols)
  KokkosSparse::spmv("N", 1.0, matrix, x, 0.0, y);
}

#if 0
template <class YType, class XType>
double dot(YType y, XType x) {
  return KokkosBlas::dot(y,x);
}

template <class ZType, class YType, class XType>
void axpby(ZType z, double alpha, XType x, double beta, YType y) {
  if(z.data() == y.data())
    return KokkosBlas::axpby(alpha,x,beta,y);
  else
    return KokkosBlas::update(alpha,x,beta,y,0.,z);
}
#endif

#else
template <class YType, class AType, class XType>
void spmv(YType y, AType A, XType x) {
  int num_rows = A.num_rows();
#ifdef USE_MKL
  mkl_dcsrgemv("N", &num_rows, A.values.data(), A.row_ptr.data(),
               A.col_idx.data(), x.data(), y.data());
#else

  // For low thread counts spread rows over individual threads
  int rows_per_team = 512;
  int team_size = 1;

  // For high concurrency architecture use teams
  if (Kokkos::DefaultExecutionSpace().concurrency() > 1024) {
    rows_per_team = 16;
    team_size = 16;
  }

  INT_TYPE nrows = y.extent(0);
  Kokkos::parallel_for(
      "SPMV",
      Kokkos::TeamPolicy<>((nrows + rows_per_team - 1) / rows_per_team,
                           team_size, 8),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
        const INT_TYPE first_row = team.league_rank() * rows_per_team;
        const INT_TYPE last_row = first_row + rows_per_team < nrows
                                      ? first_row + rows_per_team
                                      : nrows;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, first_row, last_row),
                             [&](const INT_TYPE row) {
                               const INT_TYPE row_start = A.row_ptr(row);
                               const INT_TYPE row_length =
                                   A.row_ptr(row + 1) - row_start;

                               double y_row;
                               Kokkos::parallel_reduce(
                                   Kokkos::ThreadVectorRange(team, row_length),
                                   [=](const INT_TYPE i, double &sum) {
                                     sum += A.values(i + row_start) *
                                            x(A.col_idx(i + row_start));
                                   },
                                   y_row);
                               y(row) = y_row;
                             });
      });
#endif
}
#endif

template <class YType, class XType> double dot(YType y, XType x) {
  double result;
  Kokkos::parallel_reduce(
      "DOT", y.extent(0),
      KOKKOS_LAMBDA(const INT_TYPE &i, double &lsum) { lsum += y(i) * x(i); },
      result);
  return result;
}

template <class ZType, class YType, class XType>
void axpby(ZType z, double alpha, XType x, double beta, YType y) {
  INT_TYPE n = z.extent(0);
  Kokkos::parallel_for(
      "AXPBY", n,
      KOKKOS_LAMBDA(const int &i) { z(i) = alpha * x(i) + beta * y(i); });
}

template <class VType, class AType>
int cg_solve(VType y, AType A, VType b, int max_iter, double tolerance) {
  int myproc = 0;
  int num_iters = 0;

  double normr = 0;
  double rtrans = 0;
  double oldrtrans = 0;

  INT_TYPE print_freq = max_iter / 10;
  if (print_freq > 50)
    print_freq = 50;
  if (print_freq < 1)
    print_freq = 1;
  VType x("x", b.extent(0));
  VType r("r", x.extent(0));
  VType p("r", x.extent(0));
  VType Ap("r", x.extent(0));
  double one = 1.0;
  double zero = 0.0;

  axpby(p, one, x, zero, x);
  spmv(Ap, A, p);
  axpby(r, one, b, -one, Ap);

  rtrans = dot(r, r);

  normr = std::sqrt(rtrans);

  if (myproc == 0) {
    std::cout << "Initial Residual = " << normr << std::endl;
  }

  double brkdown_tol = std::numeric_limits<double>::epsilon();

  for (INT_TYPE k = 1; k <= max_iter && normr > tolerance; ++k) {
    if (k == 1) {
      axpby(p, one, r, zero, r);
    } else {
      oldrtrans = rtrans;
      rtrans = dot(r, r);
      double beta = rtrans / oldrtrans;
      axpby(p, one, r, beta, p);
    }

    normr = std::sqrt(rtrans);

    if (myproc == 0 && (k % print_freq == 0 || k == max_iter)) {
      std::cout << "Iteration = " << k << "   Residual = " << normr
                << std::endl;
    }

    double alpha = 0;
    double p_ap_dot = 0;

    spmv(Ap, A, p);

    p_ap_dot = dot(Ap, p);

    if (p_ap_dot < brkdown_tol) {
      if (p_ap_dot < 0) {
        std::cerr << "miniFE::cg_solve ERROR, numerical breakdown!"
                  << std::endl;
        return num_iters;
      } else
        brkdown_tol = 0.1 * p_ap_dot;
    }
    alpha = rtrans / p_ap_dot;

    axpby(x, one, x, alpha, p);
    axpby(r, one, r, -alpha, Ap);
    num_iters = k;
  }
  return num_iters;
}

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);

  int N = argc > 1 ? atoi(argv[1]) : 100;
  int max_iter = argc > 2 ? atoi(argv[2]) : 200;
  double tolerance = argc > 3 ? atoi(argv[3]) : 0;

  CrsMatrix<Kokkos::HostSpace> h_A = Impl::generate_miniFE_matrix(N);
  Kokkos::View<double *, Kokkos::HostSpace> h_x =
      Impl::generate_miniFE_vector(N);

  Kokkos::View<INT_TYPE *> row_ptr("row_ptr", h_A.row_ptr.extent(0));
  Kokkos::View<INT_TYPE *> col_idx("col_idx", h_A.col_idx.extent(0));
  Kokkos::View<double *> values("values", h_A.values.extent(0));

  CrsMatrix<Kokkos::DefaultExecutionSpace::memory_space> A(
      row_ptr, col_idx, values, h_A.num_cols());
  Kokkos::View<double *> x("X", h_x.extent(0));
  Kokkos::View<double *> y("Y", h_x.extent(0));

  Kokkos::deep_copy(x, h_x);
  Kokkos::deep_copy(A.row_ptr, h_A.row_ptr);
  Kokkos::deep_copy(A.col_idx, h_A.col_idx);
  Kokkos::deep_copy(A.values, h_A.values);
  std::cout << "============" << std::endl;
  std::cout << "WarmUp Solve" << std::endl;
  std::cout << "============" << std::endl << std::endl;
  int num_iters = cg_solve(y, A, x, 20, tolerance);

  Kokkos::deep_copy(x, h_x);
  std::cout << std::endl << "============" << std::endl;
  std::cout << "Timing Solve" << std::endl;
  std::cout << "============" << std::endl << std::endl;
  Kokkos::Timer timer;
  num_iters = cg_solve(y, A, x, max_iter, tolerance);
  double time = timer.seconds();

  // Compute Bytes and Flops
  double spmv_bytes = A.num_rows() * sizeof(INT_TYPE) +
                      A.nnz() * sizeof(INT_TYPE) + A.nnz() * sizeof(double) +
                      A.nnz() * sizeof(double) + A.num_rows() * sizeof(double);

  double dot_bytes = x.extent(0) * sizeof(double) * 2;
  double axpby_bytes = x.extent(0) * sizeof(double) * 3;

  double spmv_flops = A.nnz() * 2;
  double dot_flops = x.extent(0) * 2;
  double axpby_flops = x.extent(0) * 3;

  int spmv_calls = 1 + num_iters;
  int dot_calls = num_iters * 2;
  int axpby_calls = 2 + num_iters * 3;

  printf("CGSolve for 3D (%i %i %i); %i iterations; %lf time\n", N, N, N,
         num_iters, time);
  printf(
      "Performance: %lf GFlop/s %lf GB/s (Calls SPMV: %i Dot: %i AXPBY: %i\n",
      1e-9 *
          (spmv_flops * spmv_calls + dot_flops * dot_calls +
           axpby_flops * axpby_calls) /
          time,
      (1.0 / 1024 / 1024 / 1024) *
          (spmv_bytes * spmv_calls + dot_bytes * dot_calls +
           axpby_bytes * axpby_calls) /
          time,
      spmv_calls, dot_calls, axpby_calls);
}
