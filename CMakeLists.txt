cmake_minimum_required(VERSION 3.13)
project(MurmurSIMD C)

set(CMAKE_C_STANDARD 99)

option(MurmurSIMD_DisableSIMD "Completely disables SIMD" OFF)
option(MurmurSIMD_Test "Test MurmurSIMD" OFF)

if(${MurmurSIMD_DisableSIMD})
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMURMURSIMD_DISABLE_SIMD")
endif()

# Executable
add_library(MurmurSIMD
	MurmurSIMD.c	MurmurSIMD.h
	Capabilities.c	Capabilities.h
)

# Testing
if(${MurmurSIMD_Test})
	project(MurmurSIMD_Test C)
	add_executable(MurmurSIMD_Test Tests/Test.c)
	target_link_libraries(MurmurSIMD_Test MurmurSIMD)
endif()
