#ifndef MURMURSIMD_MURMURSIMD_INTERNAL_H
#define MURMURSIMD_MURMURSIMD_INTERNAL_H

#include "Targets.h"
#include <stdint.h>
#include <string.h>

MURMURSIMD_FORCED_INLINE
void MurmurSIMD_GetBlock(const void* data, void* output, size_t blockSize)
{
	// Pointer typing
	const uint8_t* source = (const uint8_t*) data;
	uint8_t* destination = (uint8_t*) output;
	
	// Handle endian-ness
	size_t destIndex = 0;
	#ifndef MURMURSIMD_FLIP_ENDIAN
		// Little endian
		for(size_t srcIndex = 0; srcIndex < blockSize; srcIndex++)
	#else
		// Big endian
		for(size_t srcIndex = blockSize - 1; srcIndex >= 0; srcIndex--)
	#endif
	{
		destination[destIndex++] = source[srcIndex];
	}
}

#endif