cmake_minimum_required(VERSION 3.17)
project(MurmurSIMD C)

set(CMAKE_C_STANDARD 99)

# Executable
add_library(MurmurSIMD
	MurmurSIMD.h MurmurSIMD.c
	MurmurSIMD32_Scalar.c
)
