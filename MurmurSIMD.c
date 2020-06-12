#include "MurmurSIMD.h"
#include "Capabilities.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*
 * Based on this:
 * https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
 */


/*
 *   Optimal path
 */
int32_t MurmurSIMD32(const char* key, const uint32_t seed)
{
	struct MurmurSIMD_Capabilities capabilities = MurmurSIMD_GetCapabilities();
	
	#ifdef __AVX2__
	if(capabilities.AVX2) return MurmurSIMD32_AVX2(key, seed);
	#endif
	
	#ifdef __SSE2__
	if(capabilities.SSE2) return MurmurSIMD32_SSE2(key, seed);
	#endif
	
	#ifdef __MMX__
	if(capabilities.MMX) return MurmurSIMD32_MMX(key, seed);
	#endif
	
	// Fallback with no SIMD
	return MurmurSIMD32_x86(key, seed);
}

/*
 *   x86
 */
int32_t MurmurSIMD32_x86(const char* key, const uint32_t seed)
{
	const unsigned int CharsPerBlock = 4;
	
	const size_t length = strlen(key);
	const size_t blockRemainder = length % CharsPerBlock;
	const size_t blockCount = length / CharsPerBlock + (blockRemainder > 0);
	
	uint32_t hash = seed;
	
	const uint32_t c1 = 0xA329EB99;
	const uint32_t c2 = 0xBE6214AE;
	
	// Set up block buffer
	const size_t blocksSize = blockCount * sizeof(uint32_t);
	uint32_t* blocks = malloc(blocksSize + 1);
	
	// Zero the block buffer & copy the data in
	memcpy(blocks, "\0", blocksSize);
	strcpy((char*) blocks, key);
	
	for(size_t i = 0; i < blockCount; i++)
	{
		uint32_t k = blocks[i];
		
		k *= c1;
		k = k << 15;
		k |= k >> (32 - 15);
		k *= c2;
		
		hash ^= k;
		hash = hash << 13;
		hash |= hash >> (32 - 13);
		hash *= 5;
		hash += 0xE6546B64;
	}
	free(blocks);
	
	// Scramble
	hash ^= 0xCC9E2D51;
	hash = hash << 15;
	hash |= hash >> 17;
	hash *= 0x1B873593;
	
	// Finalize
	hash ^= length;
	hash ^= hash >> 16;
	hash *= 0x85EBCA6B;
	hash ^= hash >> 13;
	hash *= 0xC2B2AE35;
	hash &= hash >> 16;
	
	return hash;
}

#ifndef MURMURSIMD_DISABLE_SIMD
#include <immintrin.h>


/*
 *   MMX
 */
#ifdef __MMX__
__m64 static inline Multiply32_MMX(__m64 a, __m64 b)
{
	/*
	 * This isn't real MMX, but it's the best I can do currently.
	 */
	
	// Lower half
	const uint32_t lower = _mm_cvtsi64_si32(a) * _mm_cvtsi64_si32(b);
	
	// Shift & upper half
	a = _mm_srli_si64(a, 32);
	b = _mm_srli_si64(b, 32);
	const uint32_t upper = _mm_cvtsi64_si32(a) * _mm_cvtsi64_si32(b);
	
	return _mm_set_pi32(upper, lower);
}

__m64 static inline RotL32_MMX(const __m64 num, const int rot)
{
	const __m64 a = _mm_slli_pi32(num, rot);
	const __m64 b = _mm_srli_pi32(num, 32 - rot);
	return _mm_or_si64(a, b);
}

int32_t MurmurSIMD32_MMX(const char* key, const uint32_t seed)
{
	const unsigned int CharsPerBlock = 8;
	size_t length = strlen(key);
	
	const unsigned int remainder = length % CharsPerBlock;
	if(remainder > 0) length += remainder;
	
	// Allocate formatted data
	char* data = malloc(length);
	strcpy(data, key);
	
	__m64 hash = _mm_set1_pi32(seed);
	for(size_t i = 0; i < length; i += CharsPerBlock)
	{
		// Load data into XMM
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
#endif

/*
 *   SSE2
 */
#ifdef __SSE2__
__m128i static inline Multiply32_SSE2(const __m128i a, const __m128i b)
{
	const __m128i tmp1 = _mm_mul_epu32(a, b);	// 0, 2
	const __m128i tmp2 = _mm_mul_epu32		// 1, 3
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

__m128i static inline RotL32_SSE2(const __m128i num, const int rotation)
{
	const __m128i a = _mm_slli_epi32(num, rotation);
	const __m128i b = _mm_srli_epi32(num, 32 - rotation);
	return _mm_or_si128(a, b);
}

int32_t MurmurSIMD32_SSE2(const char* key, const uint32_t seed)
{
	const unsigned int CharsPerBlock = 16;
	size_t length = strlen(key);
	
	const unsigned int remainder = length % CharsPerBlock;
	if(remainder > 0) length += remainder;
	
	// Allocate formatted data
	char* data = malloc(length);
	strcpy(data, key);
	
	__m128i hash = _mm_set1_epi32(seed);
	for(size_t i = 0; i < length; i += CharsPerBlock)
	{
		// Load data into XMM
		__m128i k = _mm_load_si128((const __m128i*) &data[i]);
		
		k = Multiply32_SSE2
		(
			k,
			_mm_set_epi32
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
			k,
			_mm_set_epi32
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
	int32_t out = _mm_cvtsi128_si32(hash);
	for(unsigned int i = 0; i < 3; i++)
	{
		hash = _mm_srli_si128(hash, 4);
		out ^= _mm_cvtsi128_si32(hash);
	}
	return out;
}
#endif

/*
 *   AVX2
 */
#ifdef __AVX2__
__m256i static inline Multiply32_AVX2(const __m256i a, const __m256i b)
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

__m256i static inline RotL32_AVX2(const __m256i num, const int rotation)
{
	const __m256i a = _mm256_slli_epi32(num, rotation);
	const __m256i b = _mm256_srli_epi32(num, 32 - rotation);
	return _mm256_or_si256(a, b);
}

int32_t MurmurSIMD32_AVX2(const char* key, const uint32_t seed)
{
	const unsigned int CharsPerBlock = 32;
	size_t length = strlen(key);
	
	const unsigned int remainder = length % CharsPerBlock;
	if(remainder > 0) length += remainder;
	
	// Allocate formatted data
	char* data = malloc(length);
	strcpy(data, key);
	
	__m256i hash = _mm256_set1_epi32(seed);
	for(size_t i = 0; i < length; i += CharsPerBlock)
	{
		__m256i k = _mm256_load_si256((const __m256i*) &data[i]);
		
		k = Multiply32_AVX2
		(
			k,
			_mm256_set_epi32
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
			k,
			_mm256_set_epi32
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
	int32_t out = _mm256_cvtsi256_si32(hash);
	for(unsigned int i = 0; i < 7; i++)
	{
		hash = _mm256_srli_si256(hash, 4);
		out ^= _mm256_cvtsi256_si32(hash);
	}
	return out;
}
#endif

#endif