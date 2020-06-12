#ifndef MURMURSIMD_TEST_MURMURSIMD_H
#define MURMURSIMD_TEST_MURMURSIMD_H

#include <stdint.h>

int32_t MurmurSIMD32_x86(const char* key, uint32_t seed);
int32_t MurmurSIMD32_SSE2(const char* key, uint32_t seed);

#endif