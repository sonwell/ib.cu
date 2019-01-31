#pragma once
#include "memory_resource.h"
#include "scope.h"

namespace util {
	struct device {
		virtual memory_resource* memory() = 0;
	};
}
