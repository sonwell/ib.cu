#include <iostream>
#include "util/memory.h"
#include "util/allocator.h"

#include "cuda/device.h"
#include "cuda/scope.h"
#include "cuda/context.h"

void
test_scope(cuda::device& d)
{
	cuda::context context(d, cuda::stream());
	cuda::scope lock(d);
	//util::set_default_resource(d.memory());
	std::cout << d.id << " ?= " << cuda::get_device().id << std::endl;
	std::cout << d << std::endl;

	util::memory<int> vec(100);
}

int
main(void)
{
	auto& devices = cuda::devices();
	for (auto& dev: devices)
		test_scope(dev);

	return 0;
}
