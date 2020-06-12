#include "Capabilities.h"

// CPUID
#ifndef _MSC_VER

// Linux CPUID
#include <cpuid.h>
void MurmurSIMD_CPUID(int info[4], int infoType)
{
	__cpuid_count(infoType, 0, info[0], info[1], info[2], info[3]);
}

#else

// Windows CPUID
#define MurmurSIMD_CPUID(info, x) __cpuidex(info, x, NULL)

#endif

struct MurmurSIMD_Capabilities MurmurSIMD_GetCapabilities()
{
	int info[4];
	
	MurmurSIMD_CPUID(info, 0);
	const int idCount = info[0];
	
	MurmurSIMD_CPUID(info, (signed) 0x80000000);
	const unsigned int extendedIDCount = info[0];
	
	struct MurmurSIMD_Capabilities capabilities;
	
	// Populate capabilities
	#pragma clang diagnostic push
	#pragma ide diagnostic ignored "hicpp-signed-bitwise"
	
	if(idCount >= 0x00000001)
	{
		MurmurSIMD_CPUID(info, 0x00000001);
		
		capabilities.MMX	= (info[3] & (1 << 23)) != 0;
		capabilities.SSE	= (info[3] & (1 << 25)) != 0;
		capabilities.SSE2	= (info[3] & (1 << 26)) != 0;
		capabilities.SSE3	= (info[2] & (1 <<  0)) != 0;
		
		capabilities.SSSE3	= (info[2] & (1 <<  9)) != 0;
		capabilities.SSE41	= (info[2] & (1 << 19)) != 0;
		capabilities.SSE42	= (info[2] & (1 << 20)) != 0;
		capabilities.AES	= (info[2] & (1 << 25)) != 0;
		
		capabilities.AVX	= (info[2] & (1 << 28)) != 0;
		capabilities.FMA3	= (info[2] & (1 << 12)) != 0;
		
		capabilities.RDRAND	= (info[2] & (1 << 30)) != 0;
	}
	if(idCount >= 0x00000007)
	{
		MurmurSIMD_CPUID(info, 0x00000007);
		
		capabilities.AVX2			= (info[1] & (1 << 5)) != 0;
		
		capabilities.BMI1			= (info[1] & (1 <<  3)) != 0;
		capabilities.BMI2			= (info[1] & (1 <<  8)) != 0;
		capabilities.ADX			= (info[1] & (1 << 19)) != 0;
		capabilities.SHA			= (info[1] & (1 << 29)) != 0;
		capabilities.PREFETCHWT1	= (info[2] & (1 <<  0)) != 0;
		
		capabilities.AVX512F		= (info[1] & (1 << 16)) != 0;
		capabilities.AVX512CD		= (info[1] & (1 << 28)) != 0;
		capabilities.AVX512PF		= (info[1] & (1 << 26)) != 0;
		capabilities.AVX512ER		= (info[1] & (1 << 27)) != 0;
		capabilities.AVX512VL		= (info[1] & (1 << 31)) != 0;
		capabilities.AVX512BW		= (info[1] & (1 << 30)) != 0;
		capabilities.AVX512DQ		= (info[1] & (1 << 17)) != 0;
		capabilities.AVX512IFMA		= (info[1] & (1 << 21)) != 0;
		capabilities.AVX512VBMI		= (info[2] * (1 <<  1)) != 0;
	}
	if(extendedIDCount >= 0x80000001)
	{
		MurmurSIMD_CPUID(info, (signed) 0x80000001);
		
		capabilities.x64	= (info[3] & (1 << 29)) != 0;
		capabilities.ABM	= (info[2] & (1 <<  5)) != 0;
		capabilities.SSE4a	= (info[2] & (1 <<  6)) != 0;
		capabilities.FMA4	= (info[2] & (1 << 16)) != 0;
		capabilities.XOP	= (info[2] & (1 << 11)) != 0;
	}
	#pragma clang diagnostic pop
	
	return capabilities;
}

