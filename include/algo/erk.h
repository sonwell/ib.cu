#pragma once
#include <utility>

namespace algo {
namespace erk {

struct method {
	virtual int stages() const = 0;
	virtual const double* operator[](int) const = 0;
	virtual ~method() {}
};

struct midpoint : method {
private:
	static constexpr int nstages = 2;
public:
	virtual int stages() const { return nstages; }

	virtual const double*
	operator[](int n) const
	{
		static constexpr double tableau[nstages+1][nstages+1] = {
			{0.0, 0.0, 0.0},
			{0.5, 0.5, 0.0},
			{1.0, 0.0, 1.0}
		};
		return tableau[n];
	}
};

struct rk4 : method {
private:
	static constexpr int nstages = 4;
public:
	virtual int stages() const { return nstages; }

	virtual const double*
	operator[](int n) const
	{
		static constexpr double tableau[nstages+1][nstages+1] = {
			{0.0, 0.0, 0.0, 0.0, 0.0},
			{0.5, 0.5, 0.0, 0.0, 0.0},
			{0.5, 0.0, 0.5, 0.0, 0.0},
			{1.0, 0.0, 0.0, 1.0, 0.0},
			{1.0, 1./6., 1./3., 1./3., 1./6.}
		};
		return tableau[n];
	}
};

class solver {
private:
	double dt;
	std::unique_ptr<method> me;
public:
	template <typename vector_type, typename update_func, typename rhs_func>
	std::pair<vector_type, double>
	operator()(const std::pair<vector_type, double>& state, update_func axpy, rhs_func f) const
	{
		const method& m = *me;
		auto stages = m.stages();
		vector_type z[stages + 1];
		auto [y, t] = state;

		z[0] = y;
		auto s = t;
		for (int i = 0; i < stages; ++i) {
			z[i] = f(z[i], s);

			auto* weights = m[i+1];
			s = t + weights[0] * dt;
			z[i+1] = y;
			for (int j = 0; j < i+1; ++j) {
				if (!weights[1+j]) continue;
				axpy(dt * weights[1+j], z[j], z[i+1]);
			}
		}
		return {std::move(z[stages]), t + dt};
	}

	template <typename vector_type, typename rhs_func>
	std::pair<vector_type, double>
	operator()(const std::pair<vector_type, double>& state, rhs_func f) const
	{
		static constexpr auto axpy = [] (double a, const auto& x, auto& y) { y += a * x; };
		return operator()(state, axpy, f);
	}

	solver(double dt, std::unique_ptr<method> me) :
		dt(dt), me(std::move(me)) {}
	solver(double dt, method* me) :
		solver(dt, std::unique_ptr<method>(me)) {}
};

} // namespace erk
} // namespace algo
