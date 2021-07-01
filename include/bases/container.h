#pragma once
#include "util/functional.h"
#include "util/getset.h"
#include "bases/types.h"
#include "bases/operators.h"
#include "bases/geometry.h"
#include "bases/transforms.h"

namespace bases {

template <typename reference_type, typename ... shape_fns>
matrix
shape(const reference_type& ref, shape_fns ... fs)
{
	// Construct one surface per shape function, each of which transforms a
	// point in parameter space to a point in Cartesian coordinates. Some
	// possibilities are in transforms.h.
	using namespace util::functional;
	static constexpr int n = sizeof...(fs);
	static constexpr auto dims = reference_type::dimensions+1;

	static constexpr auto arrayifier = [] __host__ __device__ (auto tuple)
	{
		using tuple_type = decltype(tuple);
		constexpr auto n = std::tuple_size_v<tuple_type>;
		constexpr auto common = [] (auto t0, auto ... args) ->
			std::common_type_t<decltype(t0), decltype(args)...> { return t0; };
		using value_type = std::conditional_t<(n == 0), char,
			  decltype(apply(common, tuple))>;
		constexpr auto op = [] (auto ... args) constexpr
		{
			return std::array<value_type, n>{std::move(args)...};
		};
		return apply(op, std::move(tuple));
	};

	using seq = std::make_integer_sequence<int, n>;
	auto m = ref.num_data_sites;
	const auto& y = ref.data_geometry.position;
	matrix x{m, n * dims};

	auto* ydata = y.values();
	auto* xdata = x.values();
	auto k = [=] __device__ (int tid)
	{
		constexpr composition array{arrayifier};
		std::array<double, dims> x;
		for (int i = 0; i < dims; ++i)
			x[i] = ydata[m * i + tid];
		auto z = std::make_tuple((fs | array)(x)...);
		auto s = [&] (const auto& x, int j)
		{
			for (int i = 0; i < dims; ++i)
				xdata[n * m * i + m * j + tid] = x[i];
		};
		map(s, z, seq{});
	};
	util::transform<128, 3>(k, n ? m : 0);
	return x;
}

namespace impl {

template <typename type>
struct pair {
	type data, sample;
};

} // namespace impl

// container holds geometric information for multiple copies of the reference
// object, possibly each in different configurations.
template <typename reference_type>
struct container {
private:
	using reference_tag = decltype(reference);
	using current_tag = decltype(current);
	using operator_type = typename reference_type::operators_type;
	using geometry_type = typename reference_type::geometry_type;

	matrix& get_x() { return data.position; }

	void
	set_x(const matrix& x)
	{
		// Update geometric information when the positions are updated
		data = {ref.data_to_data, x};
		sample = {ref.data_to_sample, x};
	}
public:
	impl::pair<const operator_type&>
	operators() const
	{
		return {ref.data_to_data, ref.data_to_sample};
	}

	impl::pair<int>
	points() const
	{
		auto&& [data, sample] = operators();
		return {data.points, sample.points};
	}

	impl::pair<const geometry_type&>
	geometry(const reference_tag&) const
	{
		return {ref.data_geometry, ref.sample_geometry};
	}

	impl::pair<const geometry_type&>
	geometry(const current_tag&) const
	{
		return {data, sample};
	}

	container(const reference_type& ref) : ref{ref} {}

	template <typename ... shape_fns>
	container(const reference_type& ref, shape_fns ... fns) :
		container{ref, shape(ref, fns...)} {}

	container(const reference_type& ref, const matrix& x) :
		ref{ref} { set_x(x); }

protected:
	const reference_type& ref;
	geometry_type data;
	geometry_type sample;
public:
	util::getset<matrix&> x = {
		[&] () -> matrix& { return get_x(); },
		[&] (const matrix& x) { set_x(x); }
	};

template <typename T> friend const T& ref(const container<T>&);
};

template <typename reference_type>
inline const reference_type&
ref(const container<reference_type>& container)
{
	return container.ref;
}

} // namespace bases
