/*
 * This hash algorithm is based on this:
 * https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
 */

#ifndef MURMURSIMD_TEST_MURMURSIMD_H
#define MURMURSIMD_TEST_MURMURSIMD_H

#include <stdint.h>

/*
 *   Optimal paths
 */
int32_t MurmurSIMD32(const char* key, int32_t seed);
int64_t MurmurSIMD64(const char* key, int64_t seed);

/*
 *   32bit
 */

int32_t MurmurSIMD32_x86(const char* key, int32_t seed);

#ifndef MURMURSIMD_DISABLE_SIMD

#ifdef __MMX__
int32_t MurmurSIMD32_MMX(const char* key, int32_t seed);
#endif

#ifdef __SSE2__
int32_t MurmurSIMD32_SSE2(const char* key, int32_t seed);
#endif

#ifdef __AVX2__
int32_t MurmurSIMD32_AVX2(const char* key, uint32_t seed);
#endif

#endif	// MURMURSIMD_DISABLE_SIMD

/*
 *   64bit
 */
int64_t MurmurSIMD64_x86(const char* key, int64_t seed);

#ifndef MURMURSIMD_DISABLE_SIMD

#ifdef __SSE2__
int64_t MurmurSIMD64_SSE2(const char* key, int64_t seed);
#endif

#ifdef __AVX2__
int64_t MurmurSIMD64_AVX2(const char* key, int64_t seed);
#endif

#endif	// MURMURSIMD_DISABLE_SIMD

#endif	// HEADER