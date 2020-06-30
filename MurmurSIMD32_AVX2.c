#include "MurmurSIMD.h"

#ifndef MURMURSIMD_DISABLE_SIMD
#ifdef __AVX2__

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "Targets.h"
#include <x86intrin.h>	// AVX2

__m256i AVX2_METHOD static inline Multiply32_AVX2(const __m256i a, const __m256i b)
{
	const __m256i tmp1 = _mm256_mul_epu32(a, b);	// 0, 2, 4, 6
	const __m256i tmp2 = _mm256_mul_epu32			// 1, 3, 5, 7
	(
		_mm256_srli_si256(a, 4),
		_mm256_srli_si256(b, 4)
	);
	
	// Combine
	return _mm256_unpacklo_epi32
	(
		_mm256_shuffle_epi32(tmp1, _MM_SHUFFLE(0,0,2,0)),
		_mm256_shuffle_epi32(tmp2, _MM_SHUFFLE(0, 0, 2, 0))
	);
}

__m256i AVX2_METHOD static inline RotL32_AVX2(const __m256i num, const int rotation)
{
	const __m256i a = _mm256_slli_epi32(num, rotation);
	const __m256i b = _mm256_srli_epi32(num, 32 - rotation);
	return _mm256_or_si256(a, b);
}

int32_t AVX2_METHOD MurmurSIMD32_AVX2(const char* key, const uint32_t seed)
{
	const unsigned int CharsPerBlock = sizeof(__m256i);
	uint32_t length = strlen(key);
	
	const unsigned int remainder = length % CharsPerBlock;
	if(remainder > 0) length += remainder;
	
	// Allocate formatted data
	char* data = malloc(length + 1);
	strcpy(data, key);
	
	__m256i hash = _mm256_set1_epi32(seed);
	for(size_t i = 0; i < length; i += CharsPerBlock)
	{
		__m256i k = _mm256_load_si256((const __m256i*) &data[i]);
		
		k = Multiply32_AVX2
		(
			k, _mm256_set_epi32
			(
				(signed) 0xA329EB99,
				(signed) 0xBE6214AE,
				(signed) 0x4DC33A4D,
				(signed) 0x1FCC49A2,
				(signed) 0xC9031C00,
				(signed) 0x4093AEE0,
				(signed) 0xB33F1B01,
				(signed) 0xB19CE1AA
			)
		);
		k = RotL32_AVX2(k, 15);
		k = Multiply32_AVX2
		(
			k, _mm256_set_epi32
			(
				(signed) 0xC3059BF0,
				(signed) 0x4A99D300,
				(signed) 0x55FEC879,
				(signed) 0x82A03CCE,
				(signed) 0x91BB3DDD,
				(signed) 0x1E919393,
				(signed) 0xA39B1CE1,
				(signed) 0xDE33BB89
			)
		);
		
		hash = _mm256_xor_si256(hash, k);
		hash = RotL32_AVX2(hash, 13);
		hash = Multiply32_AVX2(hash, _mm256_set1_epi32(5));
		hash = _mm256_add_epi32(hash, _mm256_set1_epi32((signed) 0x3B11A33C));
	}
	free(data);
	
	// Finalise
	hash = _mm256_xor_si256(hash, _mm256_set1_epi32(length));
	hash = _mm256_xor_si256(hash, _mm256_srli_epi32(hash, 16));
	hash = Multiply32_AVX2(hash, _mm256_set1_epi32((signed) 0xC2AD39BB));
	hash = _mm256_xor_si256(hash, _mm256_srli_epi32(hash, 13));
	hash = Multiply32_AVX2(hash, _mm256_set1_epi32((signed) 0xE037A692));
	hash = _mm256_xor_si256(hash, _mm256_srli_epi32(hash, 16));
	
	// Convert to int
	int32_t out = 0;
	for(size_t i = 0; i < sizeof(__m256i) / sizeof(int32_t); i++)
	{
		out ^= _mm256_cvtsi256_si32(hash);
		hash = _mm256_srli_si256(hash, sizeof(int32_t));
	}
	return out;
}

#endif	// AVX2
#endif	// MURMURSIMD_DISABLE_SIMD