#include "../Capabilities.h"
#include "../MurmurSIMD.h"

#include <stdio.h>

int main()
{
	// Capability test
	printf("Capabilities:\n");
	
	const struct MurmurSIMD_Capabilities capabilities = MurmurSIMD_GetCapabilities();
	
	printf("SIMD64:\n");
	printf("\tMMX: %i\n", capabilities.MMX);
	printf("\tx64: %i\n", capabilities.x64);
	
	printf("SIMD128:\n");
	printf("\tSSE:   %i\n", capabilities.SSE);
	printf("\tSSE2:  %i\n", capabilities.SSE2);
	printf("\tSSE3:  %i\n", capabilities.SSE3);
	printf("\tSSSE3: %i\n", capabilities.SSSE3);
	printf("\tSSE41: %i\n", capabilities.SSE41);
	printf("\tSSE42: %i\n", capabilities.SSE42);
	printf("\tSSE4a: %i\n", capabilities.SSE4a);
	
	printf("SIMD256:\n");
	printf("\tAVX:  %i\n", capabilities.AVX);
	printf("\tAVX2: %i\n", capabilities.AVX2);
	printf("\tFMA3: %i\n", capabilities.FMA3);
	printf("\tFMA4: %i\n", capabilities.FMA4);
	
	printf("SIMD512:\n");
	printf("\tAVX512 Foundation: %i\n", capabilities.AVX512F);
	
	
	// Operations test
	printf("\nOperations Test:\n");
	
	const char* testString = "efzuhp|salknZzxcvjnkkkkkkkkkkkkkkkzxc;vnzxcvzkxncvkljdsfijasodfjXCKLMCZX";
	
	// 32bit
	const int32_t testSeed32 = 0xE4FCC32B;
	printf("32bit:\n\tTest String: \"%s\"\n\tTest Seed: %i\n\n", testString, testSeed32);
	printf("x86 (32bit):\t%u\n", MurmurSIMD32_x86(testString, testSeed32));
	
	#ifdef __MMX__
	printf("MMX (32bit):\t%u\n", MurmurSIMD32_MMX(testString, testSeed32));
	#endif
	
	#ifdef __SSE2__
	printf("SSE2 (32bit):\t%u\n", MurmurSIMD32_SSE2(testString, testSeed32));
	#endif
	
	#ifdef __AVX2__
	printf("AVX2 (32bit):\t%u\n", MurmurSIMD32_AVX2(testString, testSeed));
	#endif
	
	// 64 bit
	const int64_t testSeed64 = 0xCA3BDA11E4FCC32B;
	printf("\n64bit:\n\tTest String: \"%s\"\n\tTest Seed: %zi\n\n", testString, testSeed64);
	printf("x86_64:\t%zu\n", MurmurSIMD64_x86(testString, testSeed64));
	
	#ifdef __SSE2__
	printf("SSE2:\t%zu\n", MurmurSIMD64_SSE2(testString, testSeed64));
	#endif
	
	#ifdef __AVX2__
	printf("AVX2:\t%zu\n", MurmurSIMD64_AVX2(testString, testSeed64));
	#endif
}
