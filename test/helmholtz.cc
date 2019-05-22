#include <utility>
#include <iostream>
#include "lwps/base.h"
#include "lwps/matrix.h"
#include "lwps/vector.h"
#include "lwps/blas.h"

#include "fd/grid.h"
#include "fd/boundary.h"
#include "fd/dimension.h"
#include "fd/domain.h"
#include "fd/identity.h"
#include "fd/laplacian.h"
#include "fd/size.h"

#include "algo/pcg.h"
#include "algo/chebyshev.h"
#include "algo/gershgorin.h"
#include "algo/preconditioner.h"
#include "util/launch.h"
#include "cuda/event.h"

class helmholtz : public lwps::matrix_base {
private:
	double scale;
	const lwps::matrix& id;
	const lwps::matrix& lap;

	helmholtz(const lwps::matrix_size& size, double lambda, const lwps::matrix& id, const lwps::matrix& lap) :
		matrix_base(size.rows, size.cols), scale(lambda), id(id), lap(lap) {}
public:
	double lambda() const { return scale; }
	const lwps::matrix& identity() const { return id; }
	const lwps::matrix& laplacian() const { return lap; }

	helmholtz(double lambda, const lwps::matrix& id, const lwps::matrix& lap) :
		helmholtz(size(id), lambda, id, lap) {}
};

namespace lwps {
	void
	gemv(double a, const helmholtz& m, const vector& x, double b, vector& y)
	{
		if (!a) { return (void) lwps::scal(b, y); }
		const auto& id = m.identity();
		auto* values = id.values();
		auto* xvalues = x.values();
		auto* yvalues = y.values();
		auto k = [=] __device__ (int tid) {
			auto val = values[tid];
			auto scal = b * yvalues[tid];
			auto prod = a * xvalues[tid];
			if (val != 1) prod *= val;
			yvalues[tid] = scal + prod;
		};
		util::transform<128, 3>(k, id.rows());
		//gemv(a, m.identity(), x, b, y);
		gemv(a * m.lambda(), m.laplacian(), x, 1, y);
	}
}

/*inline lwps::vector
operator*(const helmholtz& m, const lwps::vector& x)
{
	std::cout << "operator*" << std::endl;
	(void) (size(m) * size(x));
	lwps::vector result{m.rows()};
	lwps::gemv(1.0, m, x, 0.0, result);
	return std::move(result);
}*/

namespace algo {
	std::pair<double, double>
	gershgorin(const helmholtz& m)
	{
		auto l = m.lambda();
		auto [a, b] = algo::gershgorin(m.identity());
		auto [c, d] = algo::gershgorin(m.laplacian());
		return {a + l * c, b + l * d};
	}
}

namespace algo {
	class chebyshev2 {
	private:
		algo::impl::chebw<4> chebw;
	protected:
		const helmholtz& op;

		lwps::vector
		polynomial(const lwps::vector& r) const
		{
			const auto& weights = chebw.weights;
			const auto& denominator = chebw.denominator;
			static constexpr auto num_weights = sizeof(weights) / sizeof(double);

			lwps::vector y = (weights[0] / denominator) * r;
			if constexpr(num_weights > 1) {
				lwps::vector z(size(r));
				for (int i = 1; i < num_weights; ++i) {
					lwps::gemv(-1, op, y, 0, z);
					lwps::axpy(weights[i] / denominator, r, z);
					lwps::swap(y, z);
				}
			}
			return std::move(y);
		}
	public:
		lwps::vector
		operator()(const lwps::vector& b) const
		{
			return polynomial(b);
		}

		chebyshev2(double a, double b, const helmholtz& m) :
			chebw(a, b), op(m) {}
	};

namespace krylov {
	lwps::vector
	pcg(const preconditioner& pr, const helmholtz& m, const lwps::vector& b, double tol)
	{
		auto&& d = solve(pr, b);
		double delta_new = dot(b, d);
		//const double eps = tol * tol;
		lwps::vector r = b;
		lwps::vector q(size(b));
		lwps::vector x(size(b), lwps::fill::zeros);
		lwps::vector s;

		int count = 0;
		while (1) {
			lwps::gemv(1.0, m, d, 0.0, q);
			auto nu = dot(d, q);
			if (nu == 0)
				break;
			auto alpha = delta_new / nu;
			lwps::axpy(alpha, d, x);
			lwps::axpy(-alpha, q, r);
			if (abs(r) <= tol)
				break;
			s = solve(pr, r);
			auto delta_old = delta_new;
			delta_new = dot(r, s);
			auto beta = delta_new / delta_old;
			lwps::axpy(beta, d, s);
			lwps::swap(d, s);
			++count;
		}
		std::cout << count << std::endl;
		return std::move(x);
	}

	inline lwps::vector
	cg(const helmholtz& m, const lwps::vector& b, double tol)
	{
		static constexpr typename algo::preconditioner::identity id;
		return pcg(id, m, b, tol);
	}
}
}


class chebyshev : public algo::preconditioner, public algo::chebyshev {
private:
	chebyshev(std::pair<double, double> range, const lwps::matrix& m) :
		algo::chebyshev(std::get<1>(range), std::get<0>(range), m) {}
public:
	virtual lwps::vector
	operator()(const lwps::vector& b) const
	{
		return algo::chebyshev::operator()(b);
	}

	chebyshev(const lwps::matrix& m) :
		chebyshev(algo::gershgorin(m), m) {}
};

class chebyshev2 : public algo::preconditioner, public algo::chebyshev2 {
private:
	chebyshev2(std::pair<double, double> range, const helmholtz& m) :
		algo::chebyshev2(std::get<1>(range), std::get<0>(range), m) {}
public:
	virtual lwps::vector
	operator()(const lwps::vector& b) const
	{
		return algo::chebyshev2::operator()(b);
	}

	chebyshev2(const helmholtz& m) :
		chebyshev2(algo::gershgorin(m), m) {}
};

int
main(void)
{
	static constexpr auto n = 128;
	static constexpr auto pi = M_PI;
	constexpr fd::dimension x{1, fd::boundary::periodic()};
	constexpr fd::dimension y{1, fd::boundary::periodic()};
	constexpr fd::dimension z{1, fd::boundary::periodic()};
	constexpr fd::domain domain{fd::grid::cell(n), x, y, z};
	constexpr auto size = fd::size(domain, x);
	constexpr auto dimensions = decltype(domain)::ndim;

	double k = 1e-5 / ((n / 128) * (n / 128));
	auto&& id = fd::identity(domain, x);
	auto&& lap = fd::laplacian(domain, x);
	auto&& hhz = - (k/2) * lap + id;
	//helmholtz hhz{-k/2, id, lap};
	chebyshev ch{hhz};

	lwps::vector b(size);
	auto* bvalues = b.values();
	auto f = [=] __device__ (int tid)
	{
		int index = tid;
		double value = 1.0;
		for (int i = 0; i < dimensions; ++i) {
			value *= cos(10 * pi * (index % n) / n);
			index /= n;
		}
		bvalues[tid] = value;
	};
	util::transform(f, size);

	for (int i = 0; i < 2; ++i) {
		cuda::event start, stop;
		start.record();
		auto&& v = algo::krylov::pcg(ch, hhz, b, 1e-8);
		stop.record();
		std::cout << (stop-start) << "ms" << std::endl;
	}


	return 0;
}
