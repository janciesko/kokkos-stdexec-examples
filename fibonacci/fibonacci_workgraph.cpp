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


#include <vector>
#include <iostream>

#include <Kokkos_Core.hpp>


/* This test is meant to be the WorkGraph equivalent of the Task DAG Scheduler
   test, please see TestTaskScheduler.hpp for that test. The algorithm computes
   the N-th fibonacci number as follows:
    - Each "task" or "work item" computes the i-th fibonacci number
    - If a task as (i < 2), it will record the known answer ahead of time.
    - If a task has (i >= 2), it will "spawn" two more tasks to compute
      the (i - 1) and (i - 2) fibonacci numbers.
      We do NOT do any de-duplication of these tasks.
      De-duplication would result in only (N - 2) tasks which must be run in
   serial. We allow duplicates both to increase the number of tasks and to
   increase the amount of available parallelism.
 */

template <class ExecSpace>
struct TestWorkGraph {
  using MemorySpace = typename ExecSpace::memory_space;
  using Policy      = Kokkos::WorkGraphPolicy<std::int32_t, ExecSpace>;
  using Graph       = typename Policy::graph_type;
  using RowMap      = typename Graph::row_map_type;
  using Entries     = typename Graph::entries_type;
  using Values      = Kokkos::View<long*, MemorySpace>;

  long m_input;
  Graph m_graph;
  Graph m_transpose;
  Values m_values;

  TestWorkGraph(long arg_input) : m_input(arg_input) {
    form_graph();
    transpose_crs(m_transpose, m_graph);
  }

  inline long full_fibonacci(long n) {
    constexpr long mask = 0x03;
    long fib[4]         = {0, 1, 1, 2};
    for (long i = 2; i <= n; ++i) {
      fib[i & mask] = fib[(i - 1) & mask] + fib[(i - 2) & mask];
    }
    return fib[n & mask];
  }

  struct HostEntry {
    long input;
    std::int32_t parent;
  };
  std::vector<HostEntry> form_host_graph() {
    std::vector<HostEntry> g;
    g.push_back({m_input, -1});
    for (std::int32_t i = 0; i < std::int32_t(g.size()); ++i) {
      auto e = g.at(std::size_t(i));
      if (e.input < 2) continue;
      /* This part of the host graph formation is the equivalent of task
         spawning in the Task DAG system. Notice how each task which is not a
         base case spawns two more tasks, without any de-duplication */
      g.push_back({e.input - 1, i});
      g.push_back({e.input - 2, i});
    }
    return g;
  }

  void form_graph() {
    auto hg = form_host_graph();
    m_graph.row_map =
        RowMap("row_map", hg.size() + 1);  // row map always has one more
    m_graph.entries =
        Entries("entries", hg.size() - 1);  // all but the first have a parent
    m_values = Values("values", hg.size());
    // printf("%zu work items\n", hg.size());
    auto h_row_map = Kokkos::create_mirror_view(m_graph.row_map);
    auto h_entries = Kokkos::create_mirror_view(m_graph.entries);
    auto h_values  = Kokkos::create_mirror_view(m_values);
    h_row_map(0)   = 0;
    for (std::int32_t i = 0; i < std::int32_t(hg.size()); ++i) {
      auto& e          = hg.at(std::size_t(i));
      h_row_map(i + 1) = i;
      if (e.input < 2) {
        h_values(i) = e.input;
      }
      if (e.parent == -1) continue;
      h_entries(i - 1) = e.parent;
    }
    Kokkos::deep_copy(m_graph.row_map, h_row_map);
    Kokkos::deep_copy(m_graph.entries, h_entries);
    Kokkos::deep_copy(m_values, h_values);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(std::int32_t i) const {
    auto begin = m_transpose.row_map(i);
    auto end   = m_transpose.row_map(i + 1);
    for (auto j = begin; j < end; ++j) {
      auto k = m_transpose.entries(j);
      m_values(i) += m_values(k);
    }
  }

  void test_for() {
    Kokkos::parallel_for(Policy(m_graph), *this);
    Kokkos::fence();
    auto h_values = Kokkos::create_mirror_view(m_values);
    Kokkos::deep_copy(h_values, m_values);
    ASSERT_EQ(h_values(0), full_fibonacci(m_input));
  }
};


int main(int argc, char **argv) {
    TestWorkGraph<TEST_EXECSPACE> f(i);
    f.test_for();
  return 0;
}