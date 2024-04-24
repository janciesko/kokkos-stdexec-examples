find_library(qthreads_lib_found qthread PATHS ${QTHREADS_ROOT} SUFFIXES lib lib64 NO_DEFAULT_PATHS)
find_path(qthreads_headers_found qthread.h PATHS ${QTHREADS_ROOT}/include NO_DEFAULT_PATHS)

find_package_handle_standard_args(QTHREADS DEFAULT_MSG qthreads_lib_found qthreads_headers_found)

if (qthreads_lib_found AND qthreads_headers_found)
  add_library(QTHREADS INTERFACE)
  set_target_properties(QTHREADS PROPERTIES
    INTERFACE_LINK_LIBRARIES ${qthreads_lib_found}
    INTERFACE_INCLUDE_DIRECTORIES ${qthreads_headers_found}
  )
endif()