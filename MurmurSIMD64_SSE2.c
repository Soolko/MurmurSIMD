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

__m128i SSE2_METHOD static inline Multiply64_SSE2(const __m128i ab, const __m128i cd)
{
	/*
	 * Based on:
	 * https://stackoverflow.com/a/54191950
	 */
	
	/*
	 * Lower half is achieved by: __m128i * (0xFFFFFFFF - 32 bit all 1s)
	 * Upper half is achieved by: __m128i >> 32
	 */
	
	// Multiply lower half of registers
	// ac = (lower half of ab) * (lower half of cd)
	__m128i ac = _mm_mul_epu32(ab, cd);
	
	// b = (upper half of ab)
	__m128i b = _mm_srli_epi64(ab, 32);
	
	// bc = b * (lower half of cd)
	__m128i bc = _mm_mul_epu32(b, cd);
	
	// d = (upper half of cd)
	__m128i d = _mm_srli_epi64(cd, 32);
	
	// ad = (upper half of ab) * d
	__m128i ad = _mm_mul_epu32(ab, d);
	
	// high = bc + ad
	__m128i high = _mm_add_epi64(bc, ad);
	
	// Move high back up to the high part of the register
	// high = high << 32
	high = _mm_slli_epi64(high, 32);
	
	// Combine lower with higher
	// Return (ac + high)
	return _mm_add_epi64(high, ac);
}

__m128i SSE2_METHOD static inline RotL64_SSE2(const __m128i num, const int rotation)
{
	const __m128i a = _mm_slli_epi64(num, rotation);
	const __m128i b = _mm_srli_epi64(num, 64 - rotation);
	return _mm_or_si128(a, b);
}

int64_t SSE2_METHOD MurmurSIMD64_SSE2(const char* key, const int64_t seed)
{
	const unsigned int CharsPerBlock = sizeof(__m128i);
	uint64_t length = strlen(key);
	
	const unsigned int remainder = length % CharsPerBlock;
	if(remainder > 0) length += remainder;
	
	// Allocate formatted data
	char* data = malloc(length + 1);
	strcpy(data, key);
	
	__m128i hash = _mm_set1_epi64x(seed);
	for(size_t i = 0; i < length; i += CharsPerBlock)
	{
		// Load data into XMM
		__m128i k = _mm_load_si128((const __m128i*) &data[i]);
		
		k = Multiply64_SSE2
		(
			k, _mm_set_epi64x
			(
				(long long) 0xC3FBA33BD33BA621,
				(long long) 0x6D3BCCC3981F33CE
			)
		);
		k = RotL64_SSE2(k, 15);
		
		k = Multiply64_SSE2
		(
			k, _mm_set_epi64x
			(
				(long long) 0x3FCAB09A08308BA1,
				(long long) 0x0B9AEE71139AB058
			)
		);
		
		hash = _mm_xor_si128(hash, k);
		hash = RotL64_SSE2(hash, 13);
		hash = Multiply64_SSE2(hash, _mm_set1_epi64x(5));
		hash = _mm_add_epi64(hash, _mm_set1_epi64x(0x34E33AB89DFAA890));
	}
	free(data);
	
	// Finalise
	hash = _mm_xor_si128(hash, _mm_set1_epi64x(length));
	hash = _mm_xor_si128(hash, _mm_srli_epi64(hash, 16));
	hash = Multiply64_SSE2(hash, _mm_set_epi64x((long long) 0xC33B0B13DAB0031A, (long long) 0xB91AA8931C3DAB3D));
	hash = _mm_xor_si128(hash, _mm_srli_epi64(hash, 13));
	hash = Multiply64_SSE2(hash, _mm_set_epi64x((long long) 0x81CD3A1897B621A3, (long long) 0x3185BAFF39A147B3));
	hash = _mm_xor_si128(hash, _mm_srli_epi64(hash, 16));
	
	// Convert to int64_t
	int64_t out = _mm_cvtsi128_si64x(hash);
	for(unsigned int i = 0; i < sizeof(int64_t) - 1; i++)
	{
		hash = _mm_srli_si128(hash, 8);
		out ^= _mm_cvtsi128_si64x(hash);
	}
	return out;
}

#endif	// SSE2
#endif	// MURMURSIMD_DISABLE_SIMD

#pragma clang diagnostic pop