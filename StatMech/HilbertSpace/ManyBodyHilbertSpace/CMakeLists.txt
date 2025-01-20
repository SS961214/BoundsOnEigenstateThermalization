include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
set(prefix "ManyBodyHilbertSpace_")

# Intel TBB is required for <execution> in ManyBodyHilbertSpace/ManyBodySpaceBase.hpp

# CPU-only codes
set(modules
	ManyBodySpinSpace
	ManyBodyBosonSpace
	ManyBodyFermionSpace
	TransSector_Spin
	ParitySector_Spin
	TransParitySector_Spin
	TransParitySector_Boson
	TransParitySector_Fermion
)
foreach(name IN LISTS modules)
	set(binary ${name}_test)
	set(target ${prefix}${binary})
	add_executable(${target} ${name}_test.cpp)
	set_target_properties(${target} PROPERTIES OUTPUT_NAME ${binary})
	target_compile_options(${target} PRIVATE -Wall -Wextra -Wfatal-errors)
	target_link_libraries(${target}
		Catch2::Catch2WithMain
		Eigen3::Eigen
		TBB::tbb
		OpenMP::OpenMP_CXX
	)
	add_test(${target} ${binary})
endforeach(name)

# CUDA/C++ codes
set(modules
	ManyBodySpinSpace
	ManyBodyBosonSpace
	ManyBodyFermionSpace
)
foreach(name IN LISTS modules)
	set(binary ${name}_onGPU_test)
	set(target ${prefix}${binary})
	add_executable(${target} ${name}_test.cu)
	set_target_properties(${target} PROPERTIES OUTPUT_NAME ${binary})
	target_compile_options(${target} PRIVATE -Wall -Wextra -Wfatal-errors)
	target_link_libraries(${target}
		Catch2::Catch2WithMain
		Eigen3::Eigen
		TBB::tbb
		OpenMP::OpenMP_CUDA
	)
	add_test(${target} compute-sanitizer --leak-check=full ${binary})
endforeach(name)