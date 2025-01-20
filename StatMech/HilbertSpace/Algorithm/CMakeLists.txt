include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
set(prefix "Algorithm_")

# CPU-only codes
set(modules
	BaseConverter
	IntegerComposition
	Combination
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
		OpenMP::OpenMP_CXX
	)
	add_test(${target} ${binary})
endforeach(name)

# CUDA/C++ codes
set(modules
	IntegerComposition
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
		OpenMP::OpenMP_CUDA
	)
	add_test(${target} compute-sanitizer --leak-check=full ${binary})
endforeach(name)