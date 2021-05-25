#ifndef MURMURSIMD_MURMURSIMD_H
#define MURMURSIMD_MURMURSIMD_H

#include <stdint.h>

uint32_t MurmurSIMD32(const void* data, size_t size, size_t typeSize);
//uint64_t MurmurSIMD64(const void* data, size_t size, size_t typeSize);

#endif