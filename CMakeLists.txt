cmake_minimum_required(VERSION 3.24)
project(mBodyETH CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
find_package(Catch2 QUIET)
if(NOT Catch2_FOUND)
	FetchContent_Declare(catch2
		GIT_REPOSITORY https://github.com/catchorg/Catch2.git
		GIT_TAG v3.1.0
	)
	FetchContent_MakeAvailable(Catch2)
endif()
enable_testing()

find_package(Eigen3 REQUIRED 3.4)
# For BLAS and LAPACK
set(MKL_INTERFACE_FULL intel_lp64) # For compatibility with Eigen
find_package(MKL REQUIRED)
find_package(TBB REQUIRED)

find_package(OpenMP REQUIRED)
execute_process(COMMAND "pkg-config" "--libs" "openmp"
	OUTPUT_VARIABLE OpenMP_LINK_LIBRARIES
	OUTPUT_STRIP_TRAILING_WHITESPACE)
set_target_properties(OpenMP::OpenMP_CXX  PROPERTIES INTERFACE_LINK_LIBRARIES "${OpenMP_LINK_LIBRARIES}")
set_target_properties(OpenMP::OpenMP_CUDA PROPERTIES INTERFACE_LINK_LIBRARIES "${OpenMP_LINK_LIBRARIES}")



find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig)
pkg_check_modules(MAGMA  REQUIRED IMPORTED_TARGET magma)

# Setup for NVCC flags
set(CMAKE_CUDA_ARCHITECTURES native)
message("-- (CUDA) CMAKE_CUDA_COMPILER_ID   = ${CMAKE_CUDA_COMPILER_ID}")
message("-- (CUDA) CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -Xcompiler -rdynamic")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall -Wextra -Wfatal-errors")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")

include_directories("${CMAKE_SOURCE_DIR}")
include_directories(SYSTEM "StatMech")
include_directories(SYSTEM "StatMech/HilbertSpace")
include_directories(SYSTEM "StatMech/MatrixGPU")


set(modules
	# OperatorRestrictionToTransSector
)
foreach(name IN LISTS modules)
	add_executable(${name} ${name}.cpp)
	target_compile_options(${name} PRIVATE -Wall -Wextra)
	target_link_libraries(${name}
		Eigen3::Eigen
		MKL::MKL
		OpenMP::OpenMP_CXX
	)
	# add_test(${name} ${name})
endforeach(name)

# add_subdirectory(noSym/CRME)
add_subdirectory(PBC_TI/ShortRange_Spin)
add_subdirectory(PBC_TI/ShortRange_Boson)
add_subdirectory(PBC_TI/ShortRange_Fermion)
add_subdirectory(PBC_TI/IsingModel)
add_subdirectory(PBC_TI/BoseHubbard)
add_subdirectory(PBC_TI/FermiHubbard)
