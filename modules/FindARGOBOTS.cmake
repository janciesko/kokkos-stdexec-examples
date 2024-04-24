find_library(argobots_lib_found libabt PATHS ${ARGOBOTS_ROOT} SUFFIXES lib lib64 NO_DEFAULT_PATHS)
find_path(argobots_headers_found abt.h PATHS ${ARGOBOTS_ROOT}/include NO_DEFAULT_PATHS)

find_package_handle_standard_args(ARGOBOTS DEFAULT_MSG argobots_lib_found argobots_headers_found)

if (argobots_lib_found AND argobots_headers_found)
  add_library(ARGOBOTS INTERFACE)
  set_target_properties(ARGOBOTS PROPERTIES
    INTERFACE_LINK_LIBRARIES ${argobots_lib_found}
    INTERFACE_INCLUDE_DIRECTORIES ${argobots_headers_found}
  )
endif()