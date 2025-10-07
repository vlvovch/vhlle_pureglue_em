cmake_minimum_required(VERSION 3.18)

project(vhlle LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# Options
option(VHLLE_WITH_ROOT "Enable ROOT-dependent code and binaries" OFF)
option(VHLLE_WITH_CUDA "Enable CUDA support when available" ON)

# Expose a path for CUDA Samples common headers (helper_cuda.h)
set(CUDA_SAMPLES_COMMON_DIR "" CACHE PATH "Path to CUDA Samples common/inc (for helper_cuda.h)")

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Detect CUDA if requested
set(VHLLE_CUDA_FOUND OFF)
if(VHLLE_WITH_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER OR CHECK_LANGUAGE_CUDA)
    enable_language(CUDA)
    set(VHLLE_CUDA_FOUND ON)
    add_definitions(-DUSE_CUDA_TOOLKIT)
  endif()
endif()

# If CUDA is on, try to find toolkit imported targets (for cudart)
if(VHLLE_CUDA_FOUND)
  find_package(CUDAToolkit QUIET)
endif()

if(VHLLE_CUDA_FOUND AND NOT CUDA_SAMPLES_COMMON_DIR)
  set(_default_cuda_samples_dir "${PROJECT_BINARY_DIR}/thirdparty/CUDA_SAMPLES")
  set(_default_helper_cuda "${_default_cuda_samples_dir}/helper_cuda.h")
  set(_default_helper_string "${_default_cuda_samples_dir}/helper_string.h")
  if(NOT EXISTS "${_default_helper_cuda}" OR NOT EXISTS "${_default_helper_string}")
    file(MAKE_DIRECTORY "${_default_cuda_samples_dir}")
    message(STATUS "helper_cuda.h not found; downloading helper headers to ${_default_cuda_samples_dir}")

    file(DOWNLOAD
      https://raw.githubusercontent.com/NVIDIA/cuda-samples/master/Common/helper_cuda.h
      "${_default_helper_cuda}"
      STATUS _download_helper_cuda_status
      TLS_VERIFY ON
    )
    list(LENGTH _download_helper_cuda_status _cuda_status_len)
    if(_cuda_status_len GREATER 0)
      list(GET _download_helper_cuda_status 0 _cuda_status_code)
      if(NOT _cuda_status_code STREQUAL "0")
        list(GET _download_helper_cuda_status 1 _cuda_status_msg)
        message(WARNING "Failed to download helper_cuda.h: ${_cuda_status_msg}")
      endif()
    endif()

    file(DOWNLOAD
      https://raw.githubusercontent.com/NVIDIA/cuda-samples/master/Common/helper_string.h
      "${_default_helper_string}"
      STATUS _download_helper_string_status
      TLS_VERIFY ON
    )
    list(LENGTH _download_helper_string_status _cuda_str_status_len)
    if(_cuda_str_status_len GREATER 0)
      list(GET _download_helper_string_status 0 _cuda_str_status_code)
      if(NOT _cuda_str_status_code STREQUAL "0")
        list(GET _download_helper_string_status 1 _cuda_str_status_msg)
        message(WARNING "Failed to download helper_string.h: ${_cuda_str_status_msg}")
      endif()
    endif()
  endif()

  if(EXISTS "${_default_helper_cuda}" AND EXISTS "${_default_helper_string}")
    set(CUDA_SAMPLES_COMMON_DIR "${_default_cuda_samples_dir}" CACHE PATH "Path to CUDA Samples common headers" FORCE)
  endif()
endif()

# Detect ROOT if requested
set(VHLLE_ROOT_FOUND OFF)
if(VHLLE_WITH_ROOT)
  # Try modern ROOT package config
  find_package(ROOT QUIET COMPONENTS Core RIO Hist Graf Graf3d Gpad Tree Rint Postscript Matrix Physics MathCore Thread)
  if(ROOT_FOUND)
    set(VHLLE_ROOT_FOUND ON)
  endif()
endif()

# Subdirectory for the core library
add_subdirectory(hlle_visc_lib)

# Example/tool: eos_calc is ROOT-less and always available
add_executable(eos_calc eos_calc/main.cpp)
target_link_libraries(eos_calc PRIVATE vhlle_visc)
target_compile_features(eos_calc PRIVATE cxx_std_17)

# Benchmarks / tests (no ROOT, no CUDA)
add_executable(bench_glauber tools/bench_glauber.cpp)
target_link_libraries(bench_glauber PRIVATE vhlle_visc)
target_compile_features(bench_glauber PRIVATE cxx_std_17)

# Optional applications
set(APP_COMMON_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/hlle_visc_lib)

add_executable(hlle_reverse hlle_reverse/main.cpp)
target_include_directories(hlle_reverse PRIVATE ${APP_COMMON_INCLUDE})
target_link_libraries(hlle_reverse PRIVATE vhlle_visc)

# if(VHLLE_CUDA_FOUND AND (CUDA_SAMPLES_COMMON_DIR OR EXISTS "$ENV{CUDA_SAMPLES_PATH}/common/inc"))
  add_executable(hlle_em hlle_em/main.cpp)
  target_include_directories(hlle_em PRIVATE ${APP_COMMON_INCLUDE})
  target_link_libraries(hlle_em PRIVATE vhlle_visc)
  if(CUDAToolkit_FOUND)
    target_link_libraries(hlle_em PRIVATE CUDA::cudart)
  endif()

  add_executable(hlle_em_pp hlle_em_pp/main.cpp hlle_em_pp/icHardSpheres.cpp)
  target_include_directories(hlle_em_pp PRIVATE ${APP_COMMON_INCLUDE})
  target_link_libraries(hlle_em_pp PRIVATE vhlle_visc)
  if(CUDAToolkit_FOUND)
    target_link_libraries(hlle_em_pp PRIVATE CUDA::cudart)
  endif()
# else()
#   message(STATUS "Skipping hlle_em and hlle_em_pp: CUDA toolkit/Samples not found or disabled")
# endif()

# ROOT-backed benchmark comparing TF1/TF2 with GL integrators
if(VHLLE_ROOT_FOUND)
  add_executable(bench_glauber_root tools/bench_glauber_root.cpp)
  if(TARGET ROOT::Core)
    target_link_libraries(bench_glauber_root PRIVATE vhlle_visc ROOT::Core ROOT::MathCore)
    target_include_directories(bench_glauber_root PRIVATE $<TARGET_PROPERTY:ROOT::Core,INTERFACE_INCLUDE_DIRECTORIES>)
  elseif(ROOT_FOUND)
    target_include_directories(bench_glauber_root PRIVATE ${ROOT_INCLUDE_DIRS})
    target_link_libraries(bench_glauber_root PRIVATE vhlle_visc ${ROOT_LIBRARIES})
  endif()
endif()
