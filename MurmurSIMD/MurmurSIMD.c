#include "MurmurSIMD.h"

#include <string.h>

uint32_t MurmurSIMD32_String(const char* data)
{
	return MurmurSIMD32(data, strlen(data), sizeof(char));
}
