#pragma clang diagnostic push
#pragma ide diagnostic ignored "hicpp-signed-bitwise"

#include "MurmurSIMD.h"

#ifndef MURMURSIMD_DISABLE_SIMD
#ifdef __SSE2__

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "Targets.h"
#include <emmintrin.h>	// SSE2

__m128i SSE2_METHOD static inline Multiply32_SSE2(const __m128i a, const __m128i b)
{
	const __m128i tmp1 = _mm_mul_epu32(a, b);	// 0, 2
	const __m128i tmp2 = _mm_mul_epu32			// 1, 3
	(
		_mm_srli_si128(a, 4),
		_mm_srli_si128(b, 4)
	);
	
	// Combine
	return _mm_unpacklo_epi32
	(
		_mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0,0,2,0)),
		_mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0, 0, 2, 0))
	);
}

__m128i SSE2_METHOD static inline RotL32_SSE2(const __m128i num, const int rotation)
{
	const __m128i a = _mm_slli_epi32(num, rotation);
	const __m128i b = _mm_srli_epi32(num, 32 - rotation);
	return _mm_or_si128(a, b);
}

int32_t SSE2_METHOD MurmurSIMD32_SSE2(const char* key, const int32_t seed)
{
	const unsigned int CharsPerBlock = sizeof(__m128i);
	uint32_t length = strlen(key);
	
	const unsigned int remainder = length % CharsPerBlock;
	if(remainder > 0) length += remainder;
	
	// Allocate formatted data
	char* data = malloc(length + 1);
	strcpy(data, key);
	
	__m128i hash = _mm_set1_epi32(seed);
	for(size_t i = 0; i < length; i += CharsPerBlock)
	{
		// Load data into XMM
		__m128i k = _mm_load_si128((const __m128i*) &data[i]);
		
		k = Multiply32_SSE2
		(
			k, _mm_set_epi32
			(
				(signed) 0xA329EB99,
				(signed) 0xBE6214AE,
				(signed) 0x4DC33A4D,
				(signed) 0x1FCC49A2
			)
		);
		k = RotL32_SSE2(k, 15);
		
		k = Multiply32_SSE2
		(
			k, _mm_set_epi32
			(
				(signed) 0xC9031C00,
				(signed) 0x4093AEE0,
				(signed) 0xB33F1B01,
				(signed) 0xB19CE1AA
			)
		);
		
		hash = _mm_xor_si128(hash, k);
		hash = RotL32_SSE2(hash, 13);
		hash = Multiply32_SSE2(hash, _mm_set1_epi32(5));
		hash = _mm_add_epi32(hash, _mm_set1_epi32((signed) 0xB4DE621A));
	}
	free(data);
	
	// Finalise
	hash = _mm_xor_si128(hash, _mm_set1_epi32(length));
	hash = _mm_xor_si128(hash, _mm_srli_epi32(hash, 16));
	hash = Multiply32_SSE2(hash, _mm_set1_epi32(0x46AB02EE));
	hash = _mm_xor_si128(hash, _mm_srli_epi32(hash, 13));
	hash = Multiply32_SSE2(hash, _mm_set1_epi32(0x7CEBBA4E));
	hash = _mm_xor_si128(hash, _mm_srli_epi32(hash, 16));
	
	// Convert to int
	int32_t out = 0;
	for(size_t i = 0; i < sizeof(__m128i) / sizeof(int32_t); i++)
	{
		out ^= _mm_cvtsi128_si32(hash);
		hash = _mm_srli_si128(hash, sizeof(int32_t));
	}
	return out;
}

#endif	// SSE2
#endif	// MURMURSIMD_DISABLE_SIMD

#pragma clang diagnostic pop