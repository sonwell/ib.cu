#pragma once
#include <memory>
#include <functional>
#include "lwps/matrix.h"
#include "lwps/vector.h"
#include "fd/domain_size.h"
#include "util/functional.h"
#include "algo/pcg.h"
#include "smoother.h"
#include "interpolation.h"

namespace mg {
	class solver;
	namespace impl {
		class direct_solver;
		class iterative_solver;

		class base_solver {
		protected:
			lwps::matrix op;

			virtual lwps::vector nested_iteration(const lwps::vector&) const = 0;
			virtual lwps::vector vcycle(const lwps::vector&) const = 0;
			virtual lwps::vector operator()(const lwps::vector&) const = 0;
			virtual lwps::vector operator()(const lwps::vector&, int) const = 0;
		public:
			virtual ~base_solver() {}
		protected:
			template <typename domain_type, typename op_func>
			base_solver(const domain_type& domain, op_func op) :
				op(op(domain)) {}
		friend class mg::solver;
		friend class direct_solver;
		friend class iterative_solver;
		};

		using solver_ptr = std::unique_ptr<base_solver>;
		using smoother_ptr = std::unique_ptr<smoother>;

		class direct_solver : public base_solver {
		protected:
			using base_solver::op;

			virtual lwps::vector nested_iteration(const lwps::vector& v) const { return operator()(v, 0); }
			virtual lwps::vector vcycle(const lwps::vector& v) const { return operator()(v, 0); }
			virtual lwps::vector operator()(const lwps::vector& v, int) const { return algo::krylov::cg(op, v, 1e-8); }
			virtual lwps::vector operator()(const lwps::vector& v) const { return operator()(v, 0); }

			template <typename domain_type, typename op_func>
			direct_solver(const domain_type& domain, op_func op) :
				base_solver(domain, op) {}
		friend class mg::solver;
		friend class iterative_solver;
		};

		class iterative_solver : public base_solver {
		private:
			smoother_ptr sm;
			lwps::matrix restriction;
			lwps::matrix interpolation;
			solver_ptr coarse;

			lwps::vector
			smooth(const lwps::vector& b) const
			{
				return solve(*sm, b);
			}

			lwps::vector
			smooth(const lwps::vector& x, const lwps::vector& b) const
			{
				return smooth(residual(x, b)) + x;
			}

			inline lwps::vector
			residual(const lwps::vector& x, const lwps::vector& b) const
			{
				/* Writing the operations in this order requires only
				 * a single buffer vector. */
				return -(op * x) + b;
			}

			template <typename pred_type>
			auto
			iterate(const lwps::vector& b, pred_type pred) const
			{
				int iteration = 0;
				auto&& fine = init(b);
				std::cout << "---" << std::endl;
				auto&& r = residual(fine, b);
				// short-circuit to avoid calling abs
				while (pred(iteration++) && abs(r) > 1e-8) {
					fine += vcycle(r);
					r = residual(fine, b);
				}
				return std::move(fine);
			}

		protected:
			using base_solver::op;

			lwps::vector
			init(const lwps::vector& b) const
			{
				auto&& restricted = restriction * b;
				auto&& iterated = coarse->nested_iteration(restricted);
				auto&& interpolated = interpolation * iterated;
				return std::move(interpolated);
			}

			virtual lwps::vector
			nested_iteration(const lwps::vector& b) const
			{
				return 0 * b;
				/*return vcycle(b);
				auto&& fine = init(b);
				auto&& r = residual(fine, b);
				std::cout << size(r) << std::endl;
				fine += vcycle(r);
				return smooth(fine, b);
				return std::move(fine);*/
			}

			virtual lwps::vector
			vcycle(const lwps::vector& b) const
			{
				auto&& fine = smooth(b);
				auto&& r = residual(fine, b);
				auto&& restricted = restriction * r;
				auto&& approx = coarse->vcycle(restricted);
				fine += interpolation * approx;
				return smooth(fine, b);
			}

			virtual lwps::vector
			operator()(const lwps::vector& b) const
			{
				return iterate(b, [] (int) { return true; });
			}

			virtual lwps::vector
			operator()(const lwps::vector& b, int its) const
			{
				return iterate(b, [=] (int it) { return it < its; });
			}

			template <typename domain_type, typename op_func, typename sm_func>
			iterative_solver(const domain_type& domain, op_func op, sm_func sm, solver_ptr&& coarse) :
				base_solver(domain, op), sm(sm(domain, base_solver::op)),
				restriction(mg::restriction(domain, std::get<0>(fd::dimensions(domain)))),
				interpolation(mg::interpolation(domain, std::get<0>(fd::dimensions(domain)))),
				coarse(std::move(coarse)) {}
			friend class mg::solver;
		};

		template <typename domain_type>
		constexpr bool
		refined(const domain_type& domain)
		{
			using namespace util::functional;
			auto k = [] (unsigned pts) { return !(pts & 1); };
			auto reduce = partial(foldl, std::logical_and<bool>(), true);
			const auto& view = std::get<0>(fd::dimensions(domain));
			return apply(reduce, map(k, fd::sizes(domain, view)));
		}
	}

	class solver {
	private:
		impl::solver_ptr slv;

		template <typename domain_type, typename op_func, typename sm_func>
		static impl::solver_ptr
		construct(const domain_type& domain, op_func op, sm_func sm)
		{
			using tag_type = typename domain_type::tag_type;
			if (impl::refined(domain)) {
				auto dims = fd::dimensions(domain);
				auto resolution = domain.resolution() >> 1;
				tag_type grid(resolution);
				auto f = [&] (auto&& ... dims) { return fd::domain{grid, dims...}; };
				auto coarse = util::functional::apply(f, dims);
				auto solver = construct(coarse, op, sm);
				return impl::solver_ptr(new impl::iterative_solver(domain, op, sm, std::move(solver)));
			}
			return impl::solver_ptr(new impl::direct_solver(domain, op));
		}

		solver&
		swap(solver& o)
		{
			std::swap(slv, o.slv);
			return *this;
		}
	public:
		lwps::vector operator()(const lwps::vector& x) const { return (*slv)(x); }
		lwps::vector operator()(const lwps::vector& x, std::size_t it) const { return (*slv)(x, it); }

		template <typename domain_type, typename op_func, typename sm_func>
		solver(const domain_type& domain, op_func op, sm_func sm) :
			slv(construct(domain, op, sm)) {}
		solver(solver&& o) : slv(nullptr) { swap(o); }
	};
}
