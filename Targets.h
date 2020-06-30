#ifndef MURMURSIMD_TARGETS_H
#define MURMURSIMD_TARGETS_H

#ifndef MURMURSIMD_DISABLE_TARGETS

#ifndef __x86_64__	// 32bit

#define x86_METHOD	__attribute__((target("arch=pentium")))
#define MMX_METHOD	__attribute__((target("arch=pentium-mmx")))
#define SSE2_METHOD	__attribute__((target("arch=pentium4")))
#define AVX2_METHOD	__attribute__((target("avx2")))

#else	// 64bit

#define x86_METHOD	// Let the compiler decide
#define MMX_METHOD	// Let the compiler decide
#define SSE2_METHOD	__attribute__((target("sse2")))
#define AVX2_METHOD	__attribute__((target("avx2")))

#endif	// x86_64

#endif	// MURMURSIMD_DISABLE_TARGETS

#endif	// Header