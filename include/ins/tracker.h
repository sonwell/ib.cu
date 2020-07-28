#pragma once
#include "cublas/iamax.h"
#include "util/cyclic_buffer.h"
#include "util/functional.h"
#include "util/log.h"
#include "simulation.h"
#include "exceptions.h"
#include "types.h"
#include "units.h"

namespace ins {

struct tracker {
private:
	units::time kmin;
	units::time kmax;
	units::time k;
	units::length h;
	units::density rho;
	util::cyclic_buffer<int, 10> buffer;

	double
	amax(const vector& v) const
	{
		cublas::handle h;
		int index;
		double value;
		cublas::iamax(h, v.rows(), v.values(), 1, &index);
		cuda::dtoh(&value, v.values() + index, 1);
		return abs(value);
	}

	template <typename v_type>
	auto
	nrminf1(const v_type& v) const
	{
		using namespace util::functional;
		auto op = [&] (const vector& v) { return amax(v); };
		return apply(partial(foldl, std::plus<void>{}, 0.0), map(op, v));
	}

	template <typename v_type>
	auto
	nrminf2(const v_type& v) const
	{
		using namespace util::functional;
		auto op = [&] (const vector& v) { auto a = amax(v); return a * a; };
		return sqrt(apply(partial(foldl, std::plus<void>{}, 0.0), map(op, v)));
	}

	int serialize(units::time k) const { return floor(log2(k/kmin)); }

	int
	current() const
	{
		int e = serialize(kmax);
		for (auto& c: buffer)
			if (c < e) e = c;
		return e;
	}

	template <typename f_type>
	int
	bound(const f_type& f)
	{
		constexpr auto eps = 1e-3;
		constexpr auto scale = 1_s / (double) 4_s;
		constexpr auto fscale = 1_kg / (1_m * 1_m * 1_s * 1_s);

		auto fmax = nrminf2(f);
		auto kmax = scale * sqrt((double) (h * rho) / (fmax + eps));
		std::cerr << "h = " << h << ", rho = " << rho << ", fmax = " << (fmax * fscale / (double) fscale) << " => kmax = " << kmax << '\n';
		return serialize(kmax);
	}
public:
	template <typename u_type, typename f_type>
	decltype(auto)
	operator()(double gamma, units::time t,
			const u_type& u, const f_type force)
	{
		using namespace util::functional;
		auto v = map([&] (const auto& w) { return gamma * w; }, u);
		int& er = *(buffer++);
		auto f = force(t + gamma * k, v);
		er = bound(f);
		while (er < current()) {
			k = kmin * (1 << er);
			f = force(t + gamma * k, v);
			er = bound(f);
		}
		auto e = current();
		if (e < 0) throw too_small();
		k = kmin * (1 << e);
		return std::pair{k, std::move(f)};
	}

	constexpr tracker(const simulation& params,
			units::length h, units::density rho) :
		kmin(params.timestep), kmax(8 * kmin), k(kmin),
		h(h), rho(rho) {}
};

}
