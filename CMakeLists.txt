cmake_minimum_required(VERSION 3.17)
project(MurmurSIMD C)

set(CMAKE_C_STANDARD 99)

# Executable
add_library(MurmurSIMD
	MurmurSIMD/Include/MurmurSIMD.h
	MurmurSIMD/Source/MurmurSIMD.c
	
	MurmurSIMD/Source/MurmurSIMD32_Scalar.c
)

target_include_directories(MurmurSIMD PUBLIC MurmurSIMD/Include PRIVATE MurmurSIMD/Source)
