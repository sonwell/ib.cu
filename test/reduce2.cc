#include "util/memory_resource.h"
#include "util/memory.h"
#include "util/launch.h"
#include "cuda/device.h"
#include "cuda/event.h"
#include "thrust/execution_policy.h"
#include "thrust/reduce.h"

static constexpr auto vt = 1;
static constexpr auto m = 1000000;
static constexpr auto n = 512000;


struct duet {
	double v[vt];

	template <typename op_type>
	constexpr __forceinline__ void
	fill(op_type&& op)
	{
		#pragma unroll
		for (int i = 0; i < vt; ++i)
			v[i] = op(i);
	}

	constexpr void
	copy(double w)
	{
		fill([&] (int) { return w; });
	}

	constexpr void
	copy(const duet& o)
	{
		fill([&] (int i) { return o.v[i]; });
	}

	constexpr duet& operator=(double w) { copy(w); return *this; }
	constexpr duet& operator=(const duet& o) { copy(o); return *this; }

	constexpr duet&
	operator+=(const duet& o)
	{
		fill([&] (int i) { return v[i] + o.v[i]; });
		return *this;
	}

	constexpr duet() : v{0} {}
	constexpr duet(double w) : duet() { copy(w); }
};

constexpr __forceinline__ duet
operator+(duet l, const duet& r)
{
	return l += r;
}

template <>
struct thrust::detail::is_arithmetic<duet> :
	thrust::detail::true_type {};

using value_type = double;

void
fill(util::memory<int>& keys)
{
	auto* kdata = keys.data();

	auto k = [=] __device__ (int tid)
	{
		kdata[tid] = tid / ((n + m - 1) / m);
	};
	util::transform(k, n);
}

struct delta : thrust::unary_function<int, value_type> {
	__host__ __device__ value_type
	operator()(int i) const
	{
		static constexpr auto pi2 = M_PI_2;
		return cos(pi2 * i / n);
		/*double dx[] = {0.25, 0.25, 0.25};

		double c[3];
		double s[3];

		for (int i = 0; i < 3; ++i) {
			c[i] = cos(pi2 * dx[i]);
			s[i] = sin(pi2 * dx[i]);
		}

		auto k = [] (int j, double s, double c)
		{
			int sign = -(j / 2);
			return 0.25 * (1 + sign * (j & 1 ? s : c));
		};

		duet r;
		for (int i = 0; i < vt; ++i) {
			double v = k(i % 4, s[0], c[0]);
			auto l = i / 4;
			for (int j = 1; j < 3; ++j) {
				v *= k(l % 4, s[j], c[j]);
				l /= 4;
			}
			r = v;
		}
		return r;*/
	}
};

int
main(void)
{
	util::set_default_resource(cuda::default_device().memory());

	util::memory<int> keys(n);
	util::memory<value_type> vout(n);
	util::memory<int> kout(n);
	util::memory<value_type> out(m);
	thrust::counting_iterator<int> cnt(0);
	thrust::transform_iterator<delta, decltype(cnt)> vdata(cnt, delta());

	fill(keys);

	thrust::device_execution_policy<thrust::system::cuda::tag> exec;
	cuda::event start, stop;
	start.record();
	auto [kend, vend] = thrust::reduce_by_key(exec, keys.data(), keys.data() + n, vdata, kout.data(), vout.data());
	//auto r = thrust::reduce(exec, vdata, vdata+n);
	stop.record();
	thrust::scatter(exec, vout.data(), vend, kout.data(), out.data());
	std::cout << (stop - start) * (64 / vt) << std::endl;

	/*
	cuda::event start, stop;
	start.record();
	algo::impl::load_store ls{values.data(), keys.data(), out.data()};
	algo::impl::reduce_op op{algo::impl::plus{}};
	algo::reduce(n, ls, op);
	stop.record();
	std::cout << (stop - start) << std::endl;
	*/

	return 0;
}
