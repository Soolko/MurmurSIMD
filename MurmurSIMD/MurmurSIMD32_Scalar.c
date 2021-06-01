#include "MurmurSIMD.h"
#include "MurmurSIMD_Internal.h"

MURMURSIMD_FORCED_INLINE
uint32_t RotL32(uint32_t x, int8_t r)
{
	return (x << r) | (x >> (32 - r));
}

/// Force all bits of the hash block to avalanche.
MURMURSIMD_FORCED_INLINE
uint32_t Mix32(uint32_t x)
{
	x ^= x >> 16;
	x *= 0x85EBCA6B;
	x ^= x >> 13;
	x *= 0XC2B2AE35;
	x ^= x >> 16;
	return x;
}

MURMURSIMD_METHOD_SCALAR
uint32_t MurmurSIMD32(const void* data, size_t size, const size_t typeSize)
{
	// Get constants
	size *= typeSize;
	const uint8_t* bytes = (const uint8_t*) data;
	
	// Magic number constants
	const uint32_t c1 = 0xCC9E2D51;
	const uint32_t c2 = 0x1B873593;
	
	// Result
	uint32_t result;
	
	// Handle blocks
	const size_t blockCount = size / sizeof(uint32_t);
	const uint32_t* blocks = (const uint32_t*) bytes;
	
	for(size_t i = 0; i < blockCount; i++)
	{
		// Get the current block as an int
		uint32_t blockResult;
		MurmurSIMD_GetBlock(&blocks[i], &blockResult, sizeof(uint32_t));
		
		blockResult *= c1;
		blockResult = RotL32(result, 15);
		blockResult *= c2;
		
		result ^= blockResult;
		result = RotL32(result, 13);
		result = result * 5 + 0xE6546B64;
	}
	
	// Handle the tail
	const uint8_t* tail = &bytes[blockCount * sizeof(uint32_t)];
	
	uint32_t tailResult = 0;
	for(size_t i = size % sizeof(uint32_t); i >= 1; i--)
	{
		tailResult ^= tail[i - 1] << ((i - 1) * 8);
	}
	tailResult *= c1;
	tailResult = RotL32(tailResult, 15);
	tailResult *= c2;
	result ^= tailResult;
	
	// Finalize
	result ^= size;
	result = Mix32(result);
	return result;
}
