cmake_minimum_required(VERSION 3.14)
project(HilbertSpace CXX)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
find_package(Catch2 QUIET)
if(NOT Catch2_FOUND)
	FetchContent_Declare(catch2
		GIT_REPOSITORY https://github.com/catchorg/Catch2.git
		GIT_TAG v3.6.0
	)
	FetchContent_MakeAvailable(Catch2)
endif()
enable_testing()

find_package(Eigen3 REQUIRED)

# For BLAS and LAPACK
set(MKL_INTERFACE_FULL intel_lp64) # For compatibility with Eigen
# set(MKL_THREADING intel_thread) # N = 10000, 10 samples, Total Test time (real) = 596.89 sec
set(MKL_THREADING gnu_thread)
find_package(MKL REQUIRED)
# message("# MKL_OMP_LIB = ${MKL_OMP_LIB}")
# message("# OMP_LIBNAME = ${OMP_LIBNAME}")
# message("# OMP_LIBRARY = ${OMP_LIBRARY}")
# message("# OMP_LINK = ${OMP_LINK}")
# message("# MKL_ENV_PATH = ${MKL_ENV_PATH}")
# message("# MKL_LINK = ${MKL_LINK}")
# message("# MKL_LINK_LINE = ${MKL_LINK_LINE}")
# message("# MKL_THREAD_LIB = ${MKL_THREAD_LIB}")
# message("# MKL_SUPP_LINK = ${MKL_SUPP_LINK}")
# get_target_property(OUT MKL::MKL INTERFACE_LINK_LIBRARIES)
# message("# MKL::MKL INTERFACE_LINK_LIBRARIES = "${OUT})

# FOR set(MKL_THREADING intel_thread)
# MKL_OMP_LIB = iomp5
# OMP_LIBNAME = libiomp5.so
# OMP_LIBRARY = /opt/intel/oneapi/compiler/2024.2/lib/libiomp5.so
# OMP_LINK = /opt/intel/oneapi/compiler/2024.2/lib/libiomp5.so
# MKL_ENV_PATH =
# MKL_LINK = dynamic
# MKL_LINK_LINE = $<IF:$<BOOL:OFF>,,>;-Wl,-rpath=$<TARGET_FILE_DIR:MKL::mkl_core>;MKL::mkl_intel_lp64;MKL::mkl_intel_thread;MKL::mkl_core
# MKL_THREAD_LIB = /opt/intel/oneapi/compiler/2024.2/lib/libiomp5.so
# MKL_SUPP_LINK = -lm;-ldl;-lpthread

# FOR set(MKL_THREADING gnu_thread)
# MKL_OMP_LIB =
# OMP_LIBNAME =
# OMP_LIBRARY =
# OMP_LINK =
# MKL_ENV_PATH =
# MKL_LINK = dynamic
# MKL_LINK_LINE = $<IF:$<BOOL:OFF>,,>;-Wl,-rpath=$<TARGET_FILE_DIR:MKL::mkl_core>;MKL::mkl_intel_lp64;MKL::mkl_gnu_thread;MKL::mkl_core
# MKL_THREAD_LIB =
# MKL_SUPP_LINK = -lgomp;-lm;-ldl;-lpthread


find_package(OpenMP REQUIRED)
# set_target_properties(OpenMP::OpenMP_CXX PROPERTIES INTERFACE_LINK_LIBRARIES "")
get_target_property(OUT OpenMP::OpenMP_CXX INTERFACE_LINK_LIBRARIES)
message(STATUS ${OUT})



set(target SelfAdjointEigenSolver_eigen_test)
add_executable(${target} ${target}.cpp)
target_compile_options(${target} PRIVATE -Wall -Wextra -Wfatal-errors)
target_link_libraries(${target}
	Catch2::Catch2WithMain
	Eigen3::Eigen
)
add_test(${target} ${target})

set(target SelfAdjointEigenSolver_eigen_openMP_test)
add_executable(${target} SelfAdjointEigenSolver_eigen_test.cpp)
target_compile_options(${target} PRIVATE -Wall -Wextra -Wfatal-errors)
target_link_libraries(${target}
	Catch2::Catch2WithMain
	Eigen3::Eigen
	OpenMP::OpenMP_CXX
)
add_test(${target} ${target})

set(target SelfAdjointEigenSolver_eigen_MKL_test)
add_executable(${target} SelfAdjointEigenSolver_eigen_test.cpp)
target_compile_options(${target} PRIVATE -Wall -Wextra -Wfatal-errors -DEIGEN_USE_MKL_ALL)
target_link_libraries(${target}
	Catch2::Catch2WithMain
	Eigen3::Eigen
	MKL::MKL
)
add_test(${target} ${target})

set(target SelfAdjointEigenSolver_eigen_MKL_OpenMP_test)
add_executable(${target} SelfAdjointEigenSolver_eigen_test.cpp)
target_compile_options(${target} PRIVATE -Wall -Wextra -Wfatal-errors -DEIGEN_USE_MKL_ALL -fopenmp)
target_link_libraries(${target}
	Catch2::Catch2WithMain
	Eigen3::Eigen
	MKL::MKL
)
add_test(${target} ${target})



find_package(PkgConfig)
pkg_check_modules(MAGMA REQUIRED IMPORTED_TARGET magma)
# PkgConfig::MAGMA needs to be linked together with MKL::MKL

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
enable_language(CUDA)
  cmake_path(GET CMAKE_SOURCE_DIR PARENT_PATH GETSMDIR)
  message("GETSMDIR = ${GETSMDIR}")
  execute_process(COMMAND "${GETSMDIR}/get_sm.sh" OUTPUT_VARIABLE DETECTED_ARCHITECTURES)
  string(REGEX REPLACE "\n" "" DETECTED_ARCHITECTURES "${DETECTED_ARCHITECTURES}")
  set(CMAKE_CUDA_ARCHITECTURES ${DETECTED_ARCHITECTURES} CACHE STRING "")
  string(REGEX REPLACE "[ \t\r\n]+" "" CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" )
  message("CMAKE_CUDA_COMPILER_ID   = ${CMAKE_CUDA_COMPILER_ID}")
  message("CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")
  set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -Xcompiler -rdynamic")

  find_package(CUDAToolkit REQUIRED)
else()
  message(STATUS "No CUDA support")
endif()

set(target SelfAdjointEigenSolver_magma_test)
add_executable(${target} SelfAdjointEigenSolver_magma_test.cu)
target_compile_options(${target} PRIVATE -Wall -Wextra -Wfatal-errors -DEIGEN_USE_MKL_ALL -fopenmp)
target_link_libraries(${target}
	Catch2::Catch2WithMain
	Eigen3::Eigen
	MKL::MKL
	PkgConfig::MAGMA
)
add_test(${target} ${target})