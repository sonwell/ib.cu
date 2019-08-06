#pragma once
#include "bases/geometry.h"
#include "types.h"
#include "load.h"

namespace forces {

struct repelling {
	template <typename object_type>
	decltype(auto)
	operator()(const object_type& object) const
	{
		using bases::current;
		const auto& curr = object.geometry(current).sample;
		loader deformed{curr};

		auto size = linalg::size(curr.position);
		auto n = size.rows * size.cols / 3;
		matrix f{size};

		auto* fdata = f.values();
		auto k = [=] __device__ (int tid)
		{
			auto curr = deformed(tid);
			auto& cx = curr.x;

			for (int i = 0; i < 3; ++i)
				fdata[i * n + tid] = 0;
		};
		util::transform(k, n);
		return f;
	}
};

} // namespace force
