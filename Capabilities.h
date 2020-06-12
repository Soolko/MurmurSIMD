#ifndef MURMURSIMD_CAPABILITIES_H
#define MURMURSIMD_CAPABILITIES_H

#include <stdbool.h>

struct MurmurSIMD_Capabilities
{
	/** MMX Support */										bool MMX;
	/** Is the CPU 64 bit? */								bool x64;
	/** Advanced Bit Manipulation */						bool ABM;
	/** Hardware random */									bool RDRAND;
															bool BMI1;
															bool BMI2;
															bool ADX;
															bool PREFETCHWT1;
	
	// SIMD128
	/** SSE Instruction set */								bool SSE;
	/** SSE2 Instruction set */								bool SSE2;
	/** SSE3 Instruction set */								bool SSE3;
	/** SSSE3 Instruction set */							bool SSSE3;
	/** SSE41 Instruction set */							bool SSE41;
	/** SSE42 Instruction set */							bool SSE42;
	/** SSE4a Instruction set */							bool SSE4a;
	/** AES HW Acceleration */								bool AES;
	/** SHA HW Acceleration */								bool SHA;
	
	// SIMD256
	/** AVX Support */										bool AVX;
															bool XOP;
	/** FMA3 Support */										bool FMA3;
	/** FMA4 Support */										bool FMA4;
	/** AVX2 Support */										bool AVX2;
	
	// SIMD512
	/** AVX512 Foundation */								bool AVX512F;
	/** AVX512 Conflict Detection */						bool AVX512CD;
	/** AVX512 Prefetch */									bool AVX512PF;
	/** AVX512 Exponential & Reciprocal */					bool AVX512ER;
	/** AVX512 Vector Length Extensions */					bool AVX512VL;
	/** AVX512 Byte & Word */								bool AVX512BW;
	/** AVX512 Double Word & Quad Word*/					bool AVX512DQ;
	/** AVX512 Integer 52-bit Fused Multiply-Add */			bool AVX512IFMA;
	/** AVX512 Vector Byte Manipulation Instructions */		bool AVX512VBMI;
};

struct MurmurSIMD_Capabilities MurmurSIMD_GetCapabilities();

#endif