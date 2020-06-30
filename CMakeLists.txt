cmake_minimum_required(VERSION 3.17)
project(MurmurSIMD C)

set(CMAKE_C_STANDARD 99)

option(MurmurSIMD_DisableSIMD "Completely disables SIMD" OFF)
option(MurmurSIMD_Test "Test MurmurSIMD" OFF)

if(${MurmurSIMD_DisableSIMD})
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMURMURSIMD_DISABLE_SIMD")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2")

# Executable
add_library(MurmurSIMD
	# Core Sources
	MurmurSIMD.h
	MurmurSIMD.c
	
	# 64bit
	MurmurSIMD64_x86.c MurmurSIMD64_SSE2.c MurmurSIMD64_AVX2.c
	
	# 32bit
	MurmurSIMD32_x86.c MurmurSIMD32_MMX.c MurmurSIMD32_SSE2.c MurmurSIMD32_AVX2.c
	
	# Extra
	Capabilities.c	Capabilities.h
)

# Testing
if(${MurmurSIMD_Test})
	project(MurmurSIMD_Test C)
	add_executable(MurmurSIMD_Test Tests/Test.c)
	target_link_libraries(MurmurSIMD_Test MurmurSIMD)
endif()
