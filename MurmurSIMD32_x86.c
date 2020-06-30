#pragma clang diagnostic push
#pragma ide diagnostic ignored "hicpp-signed-bitwise"

#include "MurmurSIMD.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int32_t static inline RotL32_x86(const int32_t num, const int rot)
{
	return (num << rot) | (num >> (32 - rot));
}

int32_t MurmurSIMD32_x86(const char* key, const int32_t seed)
{
	const unsigned int CharsPerBlock = sizeof(int32_t);
	size_t length = strlen(key);
	
	const unsigned int remainder = length % CharsPerBlock;
	if(remainder > 0) length += remainder;
	
	// Allocate formatted data
	char* data = malloc(length + 1);
	strcpy(data, key);
	
	int32_t hash = seed;
	for(size_t i = 0; i < length; i += CharsPerBlock)
	{
		// Pack chars into int
		int32_t k = *((const int32_t*) &data[i]);
		
		k *= 0xC3AE621C;
		k = RotL32_x86(k, 15);
		k *= 0x3DEAB137;
		
		hash ^= k;
		hash = RotL32_x86(hash, 13);
		hash *= 5;
		hash += 0xD18EAC57;
	}
	free(data);
	
	// Finalise
	hash ^= length;
	hash ^= hash >> 16;
	hash *= 0xAB358CA0;
	hash ^= hash >> 13;
	hash *= 0x419ACC86;
	hash ^= hash >> 16;
	return hash;
}

#pragma clang diagnostic pop