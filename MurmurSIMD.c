/*
 * This hash algorithm is based on this:
 * https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
 */

#include "MurmurSIMD.h"
#include "Capabilities.h"

/*
 * These methods pick the best available path.
 */

int32_t MurmurSIMD32(const char* key, const int32_t seed)
{
	const struct MurmurSIMD_Capabilities capabilities = MurmurSIMD_GetCapabilities();
	#ifdef __AVX2__
	if(capabilities.AVX2 && capabilities.AVX)	return MurmurSIMD32_AVX2(key, seed);
	#endif
	#ifdef __SSE2__
	if(capabilities.SSE2)	return MurmurSIMD32_SSE2(key, seed);
	#endif
	#ifdef __MMX__
	if(capabilities.MMX)	return MurmurSIMD32_MMX(key, seed);
	#endif
	return MurmurSIMD32_x86(key, seed);
}

int64_t MurmurSIMD64(const char* key, const int64_t seed)
{
	const struct MurmurSIMD_Capabilities capabilities = MurmurSIMD_GetCapabilities();
	#ifdef __AVX2__
	if(capabilities.AVX2 && capabilities.AVX)	return MurmurSIMD64_AVX2(key, seed);
	#endif
	#ifdef __SSE2__
	if(capabilities.SSE2)	return MurmurSIMD64_SSE2(key, seed);
	#endif
	return MurmurSIMD64_x86(key, seed);
}
