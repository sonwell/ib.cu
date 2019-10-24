#pragma once
#include "util/functional.h"
#include "util/math.h"
#include "algo/dot.h"
#include "algo/cross.h"

namespace bases {

template <typename ... composed>
struct composition {
	std::tuple<composed...> functions;

	template <typename arg_type>
	constexpr decltype(auto)
	operator()(arg_type&& arg) const
	{
		using namespace util::functional;
		auto op = [] (auto&& x, auto& f) { return f(x); };
		auto reduce = partial(foldl, op, arg);
		return apply(reduce, functions);
	}

	constexpr composition(std::tuple<composed...> fns) :
		functions{fns} {}
	constexpr composition(composed... fns) :
		functions{fns...} {};
};

template <typename ... composed>
composition(composed...) -> composition<composed...>;

template <typename ... composed>
composition(std::tuple<composed...>) -> composition<composed...>;

template <typename ... left, typename ... right>
decltype(auto)
operator|(const composition<left...>& f, const composition<right...>& g)
{
	return composition{std::tuple_cat(f.functions, g.functions)};
}


inline constexpr struct {
private:
	template <typename shift_type>
	constexpr decltype(auto)
	evaluate(shift_type&& dx) const
	{
		return composition{
			[=] __host__ __device__ (auto&& x)
			{
				using namespace util::functional;
				return map(std::plus<void>{}, x, dx);
			}
		};
	}
public:
	template <std::size_t n>
	constexpr decltype(auto)
	operator()(const double (&dx)[n]) const
	{
		return evaluate(dx);
	}

	template <typename shift_type>
	constexpr decltype(auto)
	operator()(shift_type&& dx) const
	{
		return evaluate(std::forward<shift_type>(dx));
	}
} translate;

inline constexpr struct {
private:
	template <typename axis_type>
	constexpr decltype(auto)
	evaluate(double angle, axis_type&& axis) const
	{
		using namespace util::functional;
		using namespace util::math;
		using algo::dot;
		using algo::cross;

		auto l = sqrt(dot(axis, axis));
		auto c = cos(angle);
		auto s = sin(angle);

		return composition{
			[=] __host__ __device__ (auto&& x)
			{
				auto d = dot(axis, x);
				auto w = cross(axis, x);
				auto f = [&] (double u, double v, double w)
				{
					return c * v + s * w / l + (1 - c) * d * u / l;
				};
				return map(f, axis, x, w);
			}
		};
	}
public:
	constexpr decltype(auto)
	operator()(double a, const double (&u)[3]) const
	{
		return evaluate(a, u);
	}

	template <typename axis_type>
	constexpr decltype(auto)
	operator()(double a, axis_type&& axis) const
	{
		return evaluate(a, std::forward<axis_type>(axis));
	}
} rotate;

} // namespace bases
