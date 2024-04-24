// @HEADER
// ***********************************************************************
//
//          Tpetra: Templated Linear Algebra Services Package
//                 Copyright (2008) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************
// @HEADER

/*
  Adapted from the Mantevo Miniapp Suite.
  https://mantevo.github.io/pdfs/MantevoOverview.pdf
*/

#ifndef GENERATE_MATRIX_HPP
#define GENERATE_MATRIX_HPP

#include <Kokkos_Core.hpp>

using INT_TYPE = int;

template <class MemSpace> struct CrsMatrix {
  Kokkos::View<INT_TYPE *, MemSpace> row_ptr;
  Kokkos::View<INT_TYPE *, MemSpace> col_idx;
  Kokkos::View<double *, MemSpace> values;

  INT_TYPE _num_cols;
  KOKKOS_INLINE_FUNCTION
  INT_TYPE num_rows() const { return row_ptr.extent(0) - 1; }
  KOKKOS_INLINE_FUNCTION
  INT_TYPE num_cols() const { return _num_cols; }
  KOKKOS_INLINE_FUNCTION
  INT_TYPE nnz() const { return values.extent(0); }

  CrsMatrix(Kokkos::View<INT_TYPE *, MemSpace> row_ptr_,
            Kokkos::View<INT_TYPE *, MemSpace> col_idx_,
            Kokkos::View<double *, MemSpace> values_, INT_TYPE num_cols_)
      : row_ptr(row_ptr_), col_idx(col_idx_), values(values_),
        _num_cols(num_cols_) {}
};

namespace Impl {
template <class GO, class S>
static void miniFE_get_row(INT_TYPE *rows, S *vals, GO *cols, INT_TYPE startrow,
                           INT_TYPE endrow, INT_TYPE &row, INT_TYPE o,
                           INT_TYPE nx1, INT_TYPE c1, INT_TYPE c2, INT_TYPE c3,
                           INT_TYPE val, INT_TYPE &miniFE_a, INT_TYPE &miniFE_b,
                           INT_TYPE &miniFE_c) {
  // FIXME (mfh 25 Jun 2014) Seriously, "val27"???  Who writes
  // code like this???

  bool val27 = false;
  if (c1 * c2 * c3 == 27) {
    val27 = true;
  }
  // printf("%li %li %li\n",c1,c2,c3);
  if ((row >= startrow) && (row < endrow)) {
    INT_TYPE offset = rows[row - startrow];
    rows[row + 1 - startrow] = offset + c1 * c2 * c3;
    for (INT_TYPE i = 0; i < c1; i++)
      for (INT_TYPE j = 0; j < c2; j++)
        for (INT_TYPE k = 0; k < c3; k++) {
          INT_TYPE m = i * c2 * c3 + j * c2 + k;
          cols[offset + m] = o + i * nx1 * nx1 + j * nx1 + k;
          if (val27) {
            bool doa = ((miniFE_a > 0) && (miniFE_a < nx1 - 3)) ||
                       ((miniFE_a == 0) && (m / 9 >= 1)) ||
                       ((miniFE_a == nx1 - 3) && (m / 9 < 2));
            bool dob = ((miniFE_b > 0) && (miniFE_b < nx1 - 3)) ||
                       ((miniFE_b == 0) && ((m % 9) / 3 >= 1)) ||
                       ((miniFE_b == nx1 - 3) && ((m % 9) / 3 < 2));
            bool doc = ((miniFE_c > 0) && (miniFE_c < nx1 - 3)) ||
                       ((miniFE_c == 0) && ((m % 3) >= 1)) ||
                       ((miniFE_c == nx1 - 3) && ((m % 3) < 2));
            if (doa && dob && doc) {
              if (m == 13)
                vals[offset + m] = 8.0 / 3.0 / (nx1 - 1);
              else {
                if (m % 2 == 1)
                  vals[offset + m] = -5.0e-1 / 3.0 / (nx1 - 1);
                else {
                  if ((m == 4) || (m == 22) || ((m > 9) && (m < 17)))
                    vals[offset + m] = -2.18960e-10 / (nx1 - 1);
                  else
                    vals[offset + m] = -2.5e-1 / 3.0 / (nx1 - 1);
                }
              }
            } else
              vals[offset + m] = 0.0;
          } else {
            if (val == m)
              vals[offset + m] = 1.0;
            else
              vals[offset + m] = 0.0;
          }
        }
  }
  if (c1 * c2 * c3 == 27) {
    miniFE_c++;
    if (miniFE_c > nx1 - 3) {
      miniFE_c = 0;
      miniFE_b++;
    }
    if (miniFE_b > nx1 - 3) {
      miniFE_b = 0;
      miniFE_a++;
    }
  }

  row++;
}

template <class GO, class S>
static void miniFE_get_block(INT_TYPE *rows, S *vals, GO *cols,
                             INT_TYPE startrow, INT_TYPE endrow, INT_TYPE &row,
                             INT_TYPE o, INT_TYPE nx1, INT_TYPE c1, INT_TYPE c2,
                             INT_TYPE val1, INT_TYPE val2, INT_TYPE val3,
                             INT_TYPE &miniFE_a, INT_TYPE &miniFE_b,
                             INT_TYPE &miniFE_c) {
  miniFE_get_row(rows, vals, cols, startrow, endrow, row, o, nx1, c1, c2, 2,
                 val1, miniFE_a, miniFE_b, miniFE_c);
  for (INT_TYPE i = 0; i < nx1 - 2; i++)
    miniFE_get_row(rows, vals, cols, startrow, endrow, row, o++, nx1, c1, c2, 3,
                   val2, miniFE_a, miniFE_b, miniFE_c);
  miniFE_get_row(rows, vals, cols, startrow, endrow, row, o++, nx1, c1, c2, 2,
                 val3, miniFE_a, miniFE_b, miniFE_c);
}

template <class GO, class S>
static void miniFE_get_superblock(INT_TYPE *rows, S *vals, GO *cols,
                                  INT_TYPE startrow, INT_TYPE endrow,
                                  INT_TYPE &row, INT_TYPE o, INT_TYPE nx1,
                                  INT_TYPE c1, INT_TYPE val1, INT_TYPE val2,
                                  INT_TYPE val3, INT_TYPE &miniFE_a,
                                  INT_TYPE &miniFE_b, INT_TYPE &miniFE_c) {
  miniFE_get_block(rows, vals, cols, startrow, endrow, row, o, nx1, c1, 2,
                   val1 + 0, val1 + val2 + 1, val1 + 1, miniFE_a, miniFE_b,
                   miniFE_c);
  for (INT_TYPE i = 0; i < nx1 - 2; i++) {
    miniFE_get_block(rows, vals, cols, startrow, endrow, row, o, nx1, c1, 3,
                     val1 + val2 + 3, val1 + val2 + val2 + val3 + 4,
                     val1 + val2 + 4, miniFE_a, miniFE_b, miniFE_c);
    o += nx1;
  }
  miniFE_get_block(rows, vals, cols, startrow, endrow, row, o, nx1, c1, 2,
                   val1 + 2, val1 + val2 + 3, val1 + 3, miniFE_a, miniFE_b,
                   miniFE_c);
}

static CrsMatrix<Kokkos::HostSpace> generate_miniFE_matrix(int nx) {
  INT_TYPE miniFE_a = 0;
  INT_TYPE miniFE_b = 0;
  INT_TYPE miniFE_c = 0;

  const int myRank = 0;
  const int numRanks = 1;

  INT_TYPE nx1 = nx + 1;

  int nrows_block = 1 + nx - 1 + 1;
  int nrows_superblock = (1 + nx - 1 + 1) * nrows_block;
  int nrows = (1 + (nx - 1) + 1) * nrows_superblock;

  INT_TYPE nnz = 0;
  nnz += 4 * (8 + (nx - 1) * 12 + 8);
  nnz += 4 * (nx - 1) * (12 + (nx - 1) * 18 + 12);
  nnz += (nx - 1) * (nx - 1) * (18 + (nx - 1) * 27 + 18);

  INT_TYPE startrow = (nrows / numRanks) * myRank;
  INT_TYPE endrow = startrow + (nrows / numRanks);
  if (endrow > nrows)
    endrow = nrows;

  Kokkos::View<INT_TYPE *, Kokkos::HostSpace> rowPtr(
      "generate_MiniFE_Matrix::rowPtr", endrow - startrow + 1);
  Kokkos::View<INT_TYPE *, Kokkos::HostSpace> colInd(
      "generate_MiniFE_Matrix::colInd", (endrow - startrow) * 27);
  Kokkos::View<double *, Kokkos::HostSpace> values(
      "generate_MiniFE_Matrix::values", (endrow - startrow) * 27);

  INT_TYPE *rows = &rowPtr[0];
  double *vals = &values[0];
  INT_TYPE *cols = &colInd[0];

  INT_TYPE row = 0;
  miniFE_get_superblock(rows, vals, cols, startrow, endrow, row, 0, nx1, 2, 0,
                        0, 0, miniFE_a, miniFE_b, miniFE_c);
  for (INT_TYPE i = 0; i < nx1 - 2; i++) {
    miniFE_get_superblock(rows, vals, cols, startrow, endrow, row,
                          i * nx1 * nx1, nx1, 3, 4, 2, 1, miniFE_a, miniFE_b,
                          miniFE_c);
  }
  miniFE_get_superblock(rows, vals, cols, startrow, endrow, row,
                        (nx1 - 2) * nx1 * nx1, nx1, 2, 4, 2, 1, miniFE_a,
                        miniFE_b, miniFE_c);

  CrsMatrix<Kokkos::HostSpace> matrix(rowPtr, colInd, values, nrows);
#ifdef USE_MKL
  for (INT_TYPE i = 0; i < rowPtr.extent(0); i++)
    rowPtr(i)++;
  for (INT_TYPE i = 0; i < colInd.extent(0); i++)
    colInd(i)++;

#endif
  return matrix;
}

template <class S>
static void miniFE_vector_generate_block(S *vec, int nx, S a, S b, int &count,
                                         int start, int end) {
  if ((count >= start) && (count < end))
    vec[count++ - start] = 0;
  for (int i = 0; i < nx - 2; i++)
    if ((count >= start) && (count < end))
      vec[count++ - start] = a / nx / nx / nx;
  if ((count >= start) && (count < end))
    vec[count++ - start] = a / nx / nx / nx + b / nx;
  if ((count >= start) && (count < end))
    vec[count++ - start] = 1;
}

template <class S>
static void miniFE_vector_generate_superblock(S *vec, int nx, S a, S b, S c,
                                              int &count, int start, int end) {
  miniFE_vector_generate_block(vec, nx, 0.0, 0.0, count, start, end);
  miniFE_vector_generate_block(vec, nx, a, b, count, start, end);
  for (int i = 0; i < nx - 3; i++)
    miniFE_vector_generate_block(vec, nx, a, c, count, start, end);
  miniFE_vector_generate_block(vec, nx, a, b, count, start, end);
  miniFE_vector_generate_block(vec, nx, 0.0, 0.0, count, start, end);
}

Kokkos::View<double *, Kokkos::HostSpace> generate_miniFE_vector(INT_TYPE nx) {
  int my_rank = 0;
  int num_ranks = 1;

  const INT_TYPE numRows = (nx + 1) * (nx + 1) * (nx + 1);

  INT_TYPE start = numRows / num_ranks * my_rank;
  INT_TYPE end = start + numRows / num_ranks;
  if (end > numRows)
    end = numRows;

  // Make a multivector X owned entirely by Proc 0.
  Kokkos::View<double *, Kokkos::HostSpace> X("X_host", numRows);
  double *vec = X.data();
  int count = 0;
  miniFE_vector_generate_superblock(vec, nx, 0.0, 0.0, 0.0, count, start, end);
  miniFE_vector_generate_superblock(vec, nx, 1.0, 5.0 / 12, 8.0 / 12, count,
                                    start, end);
  for (int i = 0; i < nx - 3; i++)
    miniFE_vector_generate_superblock(vec, nx, 1.0, 8.0 / 12, 1.0, count, start,
                                      end);
  miniFE_vector_generate_superblock(vec, nx, 1.0, 5.0 / 12, 8.0 / 12, count,
                                    start, end);
  miniFE_vector_generate_superblock(vec, nx, 0.0, 0.0, 0.0, count, start, end);

  return X;
}
} // namespace Impl
#endif
