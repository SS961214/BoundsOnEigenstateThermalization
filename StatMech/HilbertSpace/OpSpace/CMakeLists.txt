include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
set(prefix "OpSpace_")

# Intel TBB is required for <execution> in ManyBodyHilbertSpace/ManyBodySpaceBase.hpp
# Intel MKL is required for mBodyOpSpace_Boson/Fermion_GramMat

# CPU-only codes
set(modules
	OpSpace
	mBodyOpSpace_Spin
	mBodyOpSpace_Boson
	mBodyOpSpace_Fermion
	mBodyOpSpace_Boson_GramMat
	mBodyOpSpace_Fermion_GramMat
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
		MKL::MKL
		OpenMP::OpenMP_CXX
	)
	add_test(${target} ${binary})
endforeach(name)

set(modules
	show_mBodyBasisOp
	show_blocksInGramMat_Boson
)
foreach(name IN LISTS modules)
	set(binary ${name})
	set(target ${prefix}${binary})
	add_executable(${target} ${name}.cpp)
	set_target_properties(${target} PROPERTIES OUTPUT_NAME ${binary})
	target_compile_options(${target} PRIVATE -Wall -Wextra -Wfatal-errors)
	target_link_libraries(${target}
		Catch2::Catch2WithMain
		Eigen3::Eigen
		TBB::tbb
		OpenMP::OpenMP_CXX
	)
endforeach(name)


set(module mBodyOpSpace_Action)
set(types Spin Boson Fermion)
foreach(name IN LISTS modules)
	set(binary ${name}_${type})
	set(target ${prefix}${binary})
	add_executable(${target} ${name}.cpp)
	set_target_properties(${target} PROPERTIES OUTPUT_NAME ${binary})
	target_compile_options(${target} PRIVATE -Wall -Wextra -Wfatal-errors)
	target_link_libraries(${target}
		Catch2::Catch2WithMain
		Eigen3::Eigen
		TBB::tbb
		OpenMP::OpenMP_CXX
	)
endforeach(name)


# CUDA/C++ codes
set(modules
	mBodyOpSpace_Spin
	mBodyOpSpace_Boson
	mBodyOpSpace_Fermion
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
		TBB::tbb
		OpenMP::OpenMP_CUDA
	)
	add_test(${target} compute-sanitizer --leak-check=full ${binary})
endforeach(name)