#ifndef MURMURSIMD_TEST_MURMURSIMD_H
#define MURMURSIMD_TEST_MURMURSIMD_H

#include <stdint.h>

int32_t MurmurSIMD32(const char* key, uint32_t seed);

int32_t MurmurSIMD32_x86(const char* key, uint32_t seed);

#ifdef __MMX__
int32_t MurmurSIMD32_MMX(const char* key, uint32_t seed);
#endif

#ifdef __SSE2__
int32_t MurmurSIMD32_SSE2(const char* key, uint32_t seed);
#endif

#ifdef __AVX2__
int32_t MurmurSIMD32_AVX2(const char* key, uint32_t seed);
#endif

#endif