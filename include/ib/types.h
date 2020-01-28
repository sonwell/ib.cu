#pragma once
#include "util/functional.h"
#include "util/wrapper.h"
#include "util/log.h"
#include "linalg/dense.h"
#include "linalg/matrix.h"
#include "linalg/vector.h"
#include "cuda/event.h"

namespace ib {

using dense = linalg::dense<double>;
using matrix = linalg::matrix<dense>;
using vector = linalg::vector<dense>;

template <std::size_t n>
struct values_container {
	double values[n];

	constexpr double& operator[](int i) { return values[i]; }
	constexpr const double& operator[](int i) const { return values[i]; }

	constexpr values_container&
	operator+=(const values_container& o)
	{
		for (int i = 0; i < n; ++i)
			values[i] += o.values[i];
		return *this;
	}
};

template <std::size_t n>
constexpr values_container<n>
operator+(values_container<n> l, const values_container<n>& r)
{
	l += r;
	return l;
}

template <std::size_t n> using difference =
	util::wrapper<struct difference_tag, std::array<double, n>>;
template <std::size_t n> using point =
	util::wrapper<struct point_tag, std::array<double, n>>;
template <std::size_t n> using indices =
	util::wrapper<struct indices_tag, std::array<int, n>>;
template <std::size_t n> using shift =
	util::wrapper<struct shift_tag, std::array<int, n>>;

template <std::size_t dimensions>
constexpr auto
operator+(indices<dimensions> l, shift<dimensions> r)
{
	using namespace util::functional;
	map([] (int& l, const int& r) { l += r; }, l, r);
	return l;
}

struct timer {
	std::string id;
	cuda::event start, end;

	timer(std::string id) :
		id(id) { start.record(); }
	~timer()
	{
		end.record();
		util::logging::info(id, ": ", end - start, "ms");
	}
};

} // namespace ib
