# if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
#   # using Clang
# elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#   # using GCC
# elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
#   # using Intel C++
# elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
#   # using Visual Studio C++
# endif()

include(CheckCXXCompilerFlag)

check_cxx_compiler_flag(-std=c++20 CXX_HAS_CPP20)
check_cxx_compiler_flag(-Wall CXX_HAS_WALL)
check_cxx_compiler_flag(-Wextra CXX_HAS_WEXTRA)
check_cxx_compiler_flag(-Wshadow CXX_HAS_WSHADOW)
check_cxx_compiler_flag(-Wpedantic CXX_HAS_WPEDANTIC)
check_cxx_compiler_flag(-pedantic CXX_HAS_PEDANTIC)
check_cxx_compiler_flag(-Wcast-align CXX_HAS_CAST_ALIGN)
check_cxx_compiler_flag(-Wformat=2 CXX_HAS_WFORMAT2)
check_cxx_compiler_flag(-Wmissing-include-dirs CXX_HAS_WMISSING_INCLUDE_DIRS)
check_cxx_compiler_flag(-Wno-gnu-zero-variadic-macro-arguments CXX_HAS_NO_GNU_ZERO_VARIADIC_MACRO_ARGUMENTS)

add_compile_options(-Wall -Wextra -pedantic -Werror -O3)