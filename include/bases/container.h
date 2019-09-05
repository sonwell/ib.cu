#pragma once

#include "util/functional.h"
#include "util/getset.h"
#include "bases/types.h"
#include "bases/operators.h"
#include "bases/geometry.h"

namespace bases {
namespace impl {

template <typename type>
struct pair {
	type data;
	type sample;
};

template <typename reference_type>
struct base_container {
private:
	using operator_type = bases::operators<2>;
	using geometry_type = bases::geometry<2>;
	using operator_pair_type = pair<const operator_type&>;
	using geometry_pair_type = pair<const geometry_type&>;
	using reference_tag = decltype(bases::reference);
	using current_tag = decltype(bases::current);
public:
	decltype(auto)
	operators() const
	{
		using pair = operator_pair_type;
		return pair {ref.data_to_data, ref.data_to_sample};
	}

	decltype(auto)
	geometry(const reference_tag&) const
	{
		using pair = geometry_pair_type;
		return pair{ref.data_geometry, ref.sample_geometry};
	}

	decltype(auto)
	geometry(const current_tag&) const
	{
		using pair = geometry_pair_type;
		return pair{data, sample};
	}

	base_container(const reference_type& ref, const matrix& x) :
		data{ref.data_to_data, x},
		sample{ref.data_to_sample, x},
		ref{ref} {}
protected:
	geometry_type data;
	geometry_type sample;
	const reference_type& ref;
};

} // namespace impl

template <typename reference_type>
struct container : impl::base_container<reference_type> {
private:
	using base = impl::base_container<reference_type>;

	matrix& get_x() { return base::data.position; }

	void
	set_x(const matrix& x)
	{
		const auto& [data, sample] = base::operators();
		base::data = {data, x};
		base::sample = {sample, x};
	}

	template <typename ... shape_fns>
	static matrix
	shapes(const reference_type& ref, shape_fns ... fs)
	{
		using namespace util::functional;
		static constexpr int n = sizeof...(fs);
		using seq = std::make_integer_sequence<int, n>;
		auto m = ref.num_data_sites;
		const auto& y = ref.data_geometry.position;
		matrix x{m, n * 3};

		auto* ydata = y.values();
		auto* xdata = x.values();
		auto k = [=] __device__ (int tid)
		{
			std::array<double, 3> x;
			for (int i = 0; i < 3; ++i)
				x[i] = ydata[m * i + tid];
			auto z = std::make_tuple(fs(x)...);
			auto s = [&] (const auto& x, int j)
			{
				for (int i = 0; i < 3; ++i)
					xdata[n * m * i + m * j + tid] = x[i];
			};
			map(s, z, seq{});
		};
		util::transform<128, 3>(k, n ? m : 0);
		return x;
	}

public:
	template <typename ... shape_fns>
	container(const reference_type& ref, shape_fns ... fns) :
		container(ref, shapes(ref, fns...)) {}

	container(const reference_type& ref, matrix x) :
		base{ref, std::move(x)},
		x{[&] () -> matrix& { return get_x(); },
		  [&] (const matrix& x) { set_x(x); }} {}

	util::getset<matrix&> x;
};

template <std::size_t n>
constexpr decltype(auto)
shift(const double (&dx)[n])
{
	return [=] __device__ (auto x)
	{
		using namespace util::functional;
		auto s = [] (auto& x, auto&& y) { x += y; };
		map(s, x, dx);
		return x;
	};
}

template <typename shift_type>
constexpr decltype(auto)
shift(shift_type&& dx)
{
	return [=] __device__ (auto x)
	{
		using namespace util::functional;
		auto s = [] (auto& x, auto&& y) { x += y; };
		map(s, x, dx);
		return x;
	};
}

} // namespace bases
