#pragma clang diagnostic push
#pragma ide diagnostic ignored "hicpp-signed-bitwise"

#include "MurmurSIMD.h"

#ifndef MURMURSIMD_DISABLE_SIMD
#ifdef __MMX__

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "Targets.h"
#include <mmintrin.h>	// MMX

__m64 MMX_METHOD static inline Multiply32_MMX(const __m64 ab, const __m64 cd)
{
	/*
	 * Lower half is achieved by: __m64 * (0xFFFF - 16 bit all 1s)
	 * Upper half is achieved by: __m64 >> 32
	 */
	
	// Multiply lower half of registers
	// ac = (lower half of ab) * (lower half of cd)
	__m64 ac = _mm_mullo_pi16(ab, cd);
	
	// b = (upper half of ab)
	__m64 b = _mm_srli_pi32(ab, 16);
	
	// bc = b * (lower half of cd)
	__m64 bc = _mm_mullo_pi16(b, cd);
	
	// d = (upper half of cd)
	__m64 d = _mm_srli_pi32(cd, 16);
	
	// ad = (upper half of ab) * d
	__m64 ad = _mm_mulhi_pi16(ab, d);
	
	// high = bc + ad
	__m64 high = _mm_add_pi32(bc, ad);
	
	// Move high back up to the high part of the register
	high = _mm_slli_pi32(high, 16);
	
	// Combine lower with higher
	// Return (ac + high)
	return _mm_add_pi32(high, ac);
}

__m64 MMX_METHOD static inline RotL32_MMX(const __m64 num, const int rot)
{
	const __m64 a = _mm_slli_pi32(num, rot);
	const __m64 b = _mm_srli_pi32(num, 32 - rot);
	return _mm_or_si64(a, b);
}

int32_t MMX_METHOD MurmurSIMD32_MMX(const char* key, const int32_t seed)
{
	const unsigned int CharsPerBlock = sizeof(__m64);
	uint32_t length = strlen(key);
	
	const unsigned int remainder = length % CharsPerBlock;
	if(remainder > 0) length += remainder;
	
	// Allocate formatted data
	char* data = malloc(length + 1);
	strcpy(data, key);
	
	__m64 hash = _mm_set1_pi32(seed);
	for(size_t i = 0; i < length; i += CharsPerBlock)
	{
		// Load data into MMX register
		__m64 k = _mm_setr_pi8
		(
			data[i + 0], data[i + 1], data[i + 2], data[i + 3],
			data[i + 4], data[i + 5], data[i + 6], data[i + 7]
		);
		
		k = Multiply32_MMX(k, _mm_set_pi32((signed) 0xC3BBA382, (signed) 0x4EA38884));
		k = RotL32_MMX(k, 15);
		k = Multiply32_MMX(k, _mm_set_pi32((signed) 0x908A93BB, (signed) 0x083AB439));
		
		hash = _mm_xor_si64(hash, k);
		hash = RotL32_MMX(hash, 13);
		hash = Multiply32_MMX(hash, _mm_set1_pi32(5));
		hash = _mm_add_pi32(hash, _mm_set1_pi32((signed) 0xB3443E99));
	}
	free(data);
	
	// Finalise
	hash = _mm_xor_si64(hash, _mm_set1_pi32(length));
	hash = _mm_xor_si64(hash, _mm_srli_pi32(hash, 16));
	hash = Multiply32_MMX(hash, _mm_set1_pi32((signed) 0x85ECCAB3));
	hash = _mm_xor_si64(hash, _mm_srli_pi32(hash, 13));
	hash = Multiply32_MMX(hash, _mm_set1_pi32((signed) 0x900AAAE9));
	hash = _mm_xor_si64(hash, _mm_srli_pi32(hash, 16));
	
	// Convert to int
	return _mm_cvtsi64_si32(hash) ^ _mm_cvtsi64_si32(_mm_srli_si64(hash, 32));
}

#endif	// MMX
#endif	// MURMURSIMD_DISABLE_SIMD

#pragma clang diagnostic pop