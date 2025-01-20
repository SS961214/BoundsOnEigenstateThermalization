include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

set(prefix "PBCTI_BoseHubbard_")
set(modules
	quasiETHmeasure_mBody
	Hamiltonian_test
	showHamiltonian
	Hamiltonian_LSR
)

foreach(name IN LISTS modules)
	set(target ${prefix}${name})
	add_executable(${target} ${name}.cpp)
	set_target_properties(${target} PROPERTIES OUTPUT_NAME ${name})
	target_compile_options(${target} PRIVATE -Wall -Wextra)
	target_link_libraries(${target}
		Eigen3::Eigen
		MKL::MKL
		TBB::tbb
		OpenMP::OpenMP_CXX
	)
endforeach(name)

if(CMAKE_CUDA_COMPILER)
	set(modules
		quasiETHmeasure_mBody
	)
	foreach(name IN LISTS modules)
	set(target ${prefix}${name}_onGPU)
		add_executable(${target} ${name}.cu)
		set_target_properties(${target} PROPERTIES OUTPUT_NAME ${name}_onGPU)
		target_compile_options(${target} PRIVATE -Wall -Wextra -fopenmp)
		target_link_libraries(${target}
			Eigen3::Eigen
			MKL::MKL
			TBB::tbb
			PkgConfig::MAGMA
		)
	endforeach(name)
endif(CMAKE_CUDA_COMPILER)
