#pragma once
#include "util/functional.h"
#include "util/getset.h"
#include "bases/types.h"
#include "bases/operators.h"
#include "bases/geometry.h"
#include "bases/transforms.h"
#include "algo/qr.h"

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
	using operator_type = decltype(reference_type::data_to_data);
	using geometry_type = decltype(reference_type::data_geometry);
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
	points() const
	{
		auto&& [data, sample] = operators();
		return pair<int>{data.points, sample.points};
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

	base_container(const reference_type& ref) : ref{ref} {}
	base_container(const reference_type& ref, const matrix& x) :
		data{ref.data_to_data, x},
		sample{ref.data_to_sample, x},
		ref{ref} {}
protected:
	geometry_type data;
	geometry_type sample;
	const reference_type& ref;

	template <typename T> friend const T& ref(const base_container<T>&);
};

template <typename reference_type>
inline const reference_type&
ref(const base_container<reference_type>& container)
{
	return container.ref;
}

struct arrayifier {
	template <typename tuple_type>
	constexpr decltype(auto)
	operator()(const tuple_type& x) const
	{
		using namespace util::functional;
		constexpr auto n = std::tuple_size_v<tuple_type>;
		constexpr auto ct = [] (auto t0, auto ... args) ->
			std::common_type_t<decltype(t0), decltype(args)...> { return t0; };
		using array_type = std::conditional_t<(n > 0), decltype(apply(ct, x)), char>;
		constexpr auto op = [] (auto&& ... args) constexpr
		{
			return std::array<array_type, n>{args...};
		};
		return apply(op, x);
	}
};

} // namespace impl

using impl::ref;


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
	using seq = std::make_integer_sequence<int, n>;
	auto m = ref.num_data_sites;
	const auto& y = ref.data_geometry.position;
	matrix x{m, n * dims};

	auto* ydata = y.values();
	auto* xdata = x.values();
	auto k = [=] __device__ (int tid)
	{
		constexpr composition array{impl::arrayifier{}};
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

// container holds geometric information for multiple copies of the reference
// object, possibly each in different configurations.
template <typename reference_type>
struct container : impl::base_container<reference_type> {
private:
	using base = impl::base_container<reference_type>;

	matrix& get_x() { return base::sample.position; }

	void
	set_x(const matrix& x)
	{
		// Update geometric information when the positions are updated
		auto y = solve(qr, x);
		const auto& [data, sample] = base::operators();
		base::data = {data, y};
		base::sample = {sample, y};
	}

	algo::qr_factorization qr;
public:
	container(const reference_type& ref) :
		container(ref, matrix{}) {}

	template <typename ... shape_fns>
	container(const reference_type& ref, shape_fns ... fns) :
		container(ref, shape(ref, fns...)) {}

	container(const reference_type& ref, const matrix& x) :
		base{ref, x},
		qr(algo::qr(ref.data_to_sample.evaluator)),
		x{[&] () -> matrix& { return get_x(); },
		  [&] (const matrix& x) { set_x(x); }} {}

	util::getset<matrix&> x;
};

} // namespace bases
