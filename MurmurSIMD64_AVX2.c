#pragma clang diagnostic push
#pragma ide diagnostic ignored "hicpp-signed-bitwise"

#include "MurmurSIMD.h"

#ifndef MURMURSIMD_DISABLE_SIMD
#ifdef __AVX2__

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "Targets.h"
#include <immintrin.h>	// AVX2

__m256i AVX2_METHOD static inline Multiply64_AVX2(const __m256i ab, const __m256i cd)
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
	__m256i ac = _mm256_mul_epu32(ab, cd);
	
	// b = (upper half of ab)
	__m256i b = _mm256_srli_epi64(ab, 32);
	
	// bc = b * (lower half of cd)
	__m256i bc = _mm256_mul_epu32(b, cd);
	
	// d = (upper half of cd)
	__m256i d = _mm256_srli_epi64(cd, 32);
	
	// ad = (upper half of ab) * d
	__m256i ad = _mm256_mul_epu32(ab, d);
	
	// high = bc + ad
	__m256i high = _mm256_add_epi64(bc, ad);
	
	// Move high back up to the high part of the register
	// high = high << 32
	high = _mm256_slli_epi64(high, 32);
	
	// Combine lower with higher
	// Return (ac + high)
	return _mm256_add_epi64(high, ac);
}

__m256i AVX2_METHOD static inline RotL64_AVX2(const __m256i num, const int rotation)
{
	const __m256i a = _mm256_slli_epi64(num, rotation);
	const __m256i b = _mm256_srli_epi64(num, 64 - rotation);
	return _mm256_or_si256(a, b);
}

int64_t AVX2_METHOD MurmurSIMD64_AVX2(const char* key, const int64_t seed)
{
	const unsigned int CharsPerBlock = sizeof(__m256i);
	uint64_t length = strlen(key);
	
	const unsigned int remainder = length % CharsPerBlock;
	if(remainder > 0) length += remainder;
	
	// Allocate formatted data
	char* data = malloc(length + 1);
	strcpy(data, key);
	
	__m256i hash = _mm256_set1_epi64x(seed);
	for(size_t i = 0; i < length; i += CharsPerBlock)
	{
		// Load data into XMM
		__m256i k = _mm256_load_si256((const __m256i*) &data[i]);
		
		k = Multiply64_AVX2
		(
			k, _mm256_set_epi64x
			(
				(long long) 0xC3FBA33BD33BA621,
				(long long) 0x6D3BCCC3981F33CE,
				(long long) 0xBA1B0AC3AB867D38,
				(long long) 0x39BDE3EE19A8BA27
			)
		);
		k = RotL64_AVX2(k, 15);
		
		k = Multiply64_AVX2
		(
			k, _mm256_set_epi64x
			(
				(long long) 0x3FCAB09A08308BA1,
				(long long) 0x0B9AEE71139AB058,
				(long long) 0x9C210AE3ABB91073,
				(long long) 0xC9A1BA38A0B3FEDD
			)
		);
		
		hash = _mm256_xor_si256(hash, k);
		hash = RotL64_AVX2(hash, 13);
		hash = Multiply64_AVX2(hash, _mm256_set1_epi64x(5));
		hash = _mm256_add_epi64(hash, _mm256_set1_epi64x((long long) 0xB3A1A55DEF917A7C));
	}
	free(data);
	
	// Finalise
	hash = _mm256_xor_si256(hash, _mm256_set1_epi64x(length));
	hash = _mm256_xor_si256(hash, _mm256_srli_epi64(hash, 16));
	hash = Multiply64_AVX2
	(
		hash, _mm256_set_epi64x
		(
			(long long) 0xC33B0B13DAB0031A,
			(long long) 0xB91AA8931C3DAB3D,
			(long long) 0x6C3E39A1657919A3,
			(long long) 0x2AC4FBE92A76C577
		)
	);
	hash = _mm256_xor_si256(hash, _mm256_srli_epi64(hash, 13));
	hash = Multiply64_AVX2
	(
		hash, _mm256_set_epi64x
		(
			(long long) 0x81CD3A1897B621A3,
			(long long) 0x3185BAFF39A147B3,
			(long long) 0xDA376BA7CDEA8C7B,
			(long long) 0xB9A7C56EFD4FBCAB
		)
	);
	hash = _mm256_xor_si256(hash, _mm256_srli_epi64(hash, 16));
	
	// Convert to int64_t
	int64_t out = 0;
	for(size_t i = 0; i < sizeof(__m256i) / sizeof(int64_t); i++)
	{
		int64_t current = _mm256_cvtsi256_si32(hash);
		
		hash = _mm256_srli_si256(hash, 4);
		out |= (int64_t) _mm256_cvtsi256_si32(hash) << 32;
		
		out ^= current;
	}
	return out;
}

#endif	// AVX2
#endif	// MURMURSIMD_DISABLE_SIMD

#pragma clang diagnostic pop