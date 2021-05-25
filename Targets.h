#ifndef MURMURSIMD_TARGETS_H
#define MURMURSIMD_TARGETS_H

// Inlining
#define MURMURSIMD_FORCED_INLINE inline __attribute__((always_inline))

// SIMD
#define MURMURSIMD_METHOD_SCALAR __attribute__((target("default")))
#define MURMURSIMD_METHOD_MMX    __attribute__((target("mmx")))
#define MURMURSIMD_METHOD_SSE2   __attribute__((target("sse2")))
#define MURMURSIMD_METHOD_AVX2   __attribute__((target("avx2")))

// Set big endian if other libraries are active
#ifdef SDL_BIG_ENDIAN
#define MURMURSIMD_FLIP_ENDIAN
#endif

#endif