#pragma clang diagnostic push
#pragma ide diagnostic ignored "hicpp-signed-bitwise"

#include "MurmurSIMD.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int64_t static inline RotL64_x64(const int64_t num, const int rot)
{
	return (num << rot) | (num >> (64 - rot));
}

int64_t MurmurSIMD64_x86(const char* key, int64_t seed)
{
	const unsigned int CharsPerBlock = sizeof(int64_t);
	size_t length = strlen(key);
	
	const unsigned int remainder = length % CharsPerBlock;
	if(remainder > 0) length += remainder;
	
	// Allocate formatted data
	char* data = malloc(length + 1);
	strcpy(data, key);
	
	int64_t hash = seed;
	for(size_t i = 0; i < length; i += CharsPerBlock)
	{
		// Pack chars into int
		int64_t k = *((const int64_t*) &key[i]);
		
		k *= 0xFC340BE1DD381CBBl;
		k = RotL64_x64(k, 15);
		k *= 0xC375A1BB3CDA1011l;
		
		hash ^= k;
		hash = RotL64_x64(hash, 13);
		hash *= 5;
		hash += 0x78BA4EC69BAAEl;
	}
	free(data);
	
	// Finalise
	hash ^= length;
	hash ^= hash >> 16;
	hash *= 0xC32BA869A6CE7BDEl;
	hash ^= hash >> 13;
	hash *= 0x3ECB791DB93621FFl;
	hash ^= hash >> 16;
	return hash;
}

#pragma clang diagnostic pop