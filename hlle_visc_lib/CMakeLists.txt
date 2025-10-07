set(_lib_src_root ${CMAKE_CURRENT_SOURCE_DIR})

# Core sources (no ROOT, no CUDA)
set(VHLLE_CORE_SOURCES
  ${_lib_src_root}/NumericalIntegration.cpp
  ${_lib_src_root}/cll.cpp
  ${_lib_src_root}/cornelius.cpp
  ${_lib_src_root}/eos.cpp
  ${_lib_src_root}/eo1.cpp
  ${_lib_src_root}/eo3.cpp
  ${_lib_src_root}/eoAZH.cpp
  ${_lib_src_root}/eoChiral.cpp
  ${_lib_src_root}/eoHadron.cpp
  ${_lib_src_root}/eoSimpleSpline.cpp
  ${_lib_src_root}/eosLinearCombination.cpp
  ${_lib_src_root}/eosLinearCombinationTable.cpp
  ${_lib_src_root}/eosuQGP.cpp
  ${_lib_src_root}/fld.cpp
  ${_lib_src_root}/hdo.cpp
  ${_lib_src_root}/ic.cpp
  ${_lib_src_root}/icBjorken.cpp
  ${_lib_src_root}/icGlauber.cpp
  ${_lib_src_root}/icGlauberMC.cpp
  ${_lib_src_root}/icGubser.cpp
  ${_lib_src_root}/icPartUrqmd.cpp
  ${_lib_src_root}/ickw.cpp
  ${_lib_src_root}/icurqmd.cpp
  ${_lib_src_root}/dileptons.cpp
  ${_lib_src_root}/photons.cpp
  ${_lib_src_root}/rmn.cpp
  ${_lib_src_root}/s95p.cpp
  ${_lib_src_root}/trancoeff.cpp
  ${_lib_src_root}/photonsCUDA.cpp
)

# ROOT-dependent sources (guarded)
set(VHLLE_ROOT_SOURCES)

# CUDA sources (guarded)
set(VHLLE_CUDA_CU  ${_lib_src_root}/photonsCUDA.cu)

set(VHLLE_ALL_SOURCES ${VHLLE_CORE_SOURCES})

if(VHLLE_ROOT_FOUND)
  list(APPEND VHLLE_ALL_SOURCES ${VHLLE_ROOT_SOURCES})
endif()

set(_use_cuda_sources OFF)
if(VHLLE_CUDA_FOUND)
  # Require helper_cuda.h location either via explicit var or typical env path
  if(CUDA_SAMPLES_COMMON_DIR AND EXISTS ${CUDA_SAMPLES_COMMON_DIR}/helper_cuda.h)
    set(_use_cuda_sources ON)
  elseif(DEFINED ENV{CUDA_SAMPLES_PATH} AND EXISTS $ENV{CUDA_SAMPLES_PATH}/common/inc/helper_cuda.h)
    set(CUDA_SAMPLES_COMMON_DIR "$ENV{CUDA_SAMPLES_PATH}/common/inc")
    set(_use_cuda_sources ON)
  elseif(EXISTS "/usr/local/cuda/samples/common/inc/helper_cuda.h")
    set(CUDA_SAMPLES_COMMON_DIR "/usr/local/cuda/samples/common/inc")
    set(_use_cuda_sources ON)
  endif()
endif()

if(_use_cuda_sources)
  list(APPEND VHLLE_ALL_SOURCES ${VHLLE_CUDA_CPP} ${VHLLE_CUDA_CU})
endif()

add_library(vhlle_visc STATIC ${VHLLE_ALL_SOURCES})
target_include_directories(vhlle_visc
  PUBLIC
    ${_lib_src_root}
)

# CUDA compile settings
if(_use_cuda_sources)
  set_target_properties(vhlle_visc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  # CUDA Samples helper includes
  target_include_directories(vhlle_visc PRIVATE ${CUDA_SAMPLES_COMMON_DIR})
endif()

# On MSVC, match original project definitions to avoid warnings
if(MSVC)
  target_compile_definitions(vhlle_visc PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# Root includes for library only when building root-dependent sources
if(VHLLE_ROOT_FOUND)
  if(TARGET ROOT::Core)
    target_link_libraries(vhlle_visc PRIVATE ROOT::Core)
    target_include_directories(vhlle_visc PRIVATE $<TARGET_PROPERTY:ROOT::Core,INTERFACE_INCLUDE_DIRECTORIES>)
  elseif(ROOT_FOUND)
    target_include_directories(vhlle_visc PRIVATE ${ROOT_INCLUDE_DIRS})
  endif()
endif()

target_compile_features(vhlle_visc PUBLIC cxx_std_17)
