cmake_minimum_required(VERSION 3.24)
project(StatMech CXX CUDA)
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
include_directories(SYSTEM "HilbertSpace")
include_directories(SYSTEM "MatrixGPU")


# CPU-only codes
set(modules
	quasiETHmeasure_Spin
	distributeBlocks
)
foreach(name IN LISTS modules)
	set(binary ${name}_test)
	set(target ${binary})
	add_executable(${target} ${name}_test.cpp)
	set_target_properties(${target} PROPERTIES OUTPUT_NAME ${binary})
	target_compile_options(${target} PRIVATE -Wall -Wextra -Wfatal-errors)
	target_link_libraries(${target}
		Catch2::Catch2WithMain
		Eigen3::Eigen
		MKL::MKL
		TBB::tbb
		OpenMP::OpenMP_CXX
	)
	add_test(${target} ${binary})
endforeach(name)


# CUDA/C++ codes
set(modules
	CholeskyFactorization
	calculateEEV
	calculateGramMat
	quasiETHmeasure_Spin
	quasiETHmeasure_Boson
	quasiETHmeasure_Fermion
)
foreach(name IN LISTS modules)
	set(binary ${name}_onGPU_test)
	set(target ${binary})
	add_executable(${target} ${name}_test.cu)
	set_target_properties(${target} PROPERTIES OUTPUT_NAME ${binary})
	target_compile_options(${target} PRIVATE -Wall -Wextra -Wfatal-errors)
	target_link_libraries(${target}
		Catch2::Catch2WithMain
		Eigen3::Eigen
		MKL::MKL
		TBB::tbb
		PkgConfig::MAGMA
	)
	add_test(${target} compute-sanitizer --leak-check=full ${binary})
endforeach(name)


set(modules
	# quasiETHmeasure_Spin
	quasiETHmeasure_Boson
	quasiETHmeasure_Fermion
)
foreach(name IN LISTS modules)
	set(target ${name}_onGPU_profile)
	add_executable(${target} ${name}_profile.cu)
	target_compile_options(${target} PRIVATE -Wall -Wextra -Wfatal-errors)
	target_link_libraries(${target}
		Eigen3::Eigen
		MKL::MKL
		TBB::tbb
		PkgConfig::MAGMA
	)
endforeach(name)