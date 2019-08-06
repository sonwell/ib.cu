#pragma once
#include "util/memory.h"
#include "util/launch.h"
#include "types.h"
#include "base.h"

namespace linalg {

template <typename> class dense;
template <typename vtype, bool = is_scalar_v<vtype>> class filler;

template <typename vtype>
class dense : public base {
private:
	static_assert(is_field_v<vtype>, "type is not double, float, or complex");
public:
	using value_type = vtype;
	using index_type = typename base::index_type;
protected:
	using value_ptr = util::memory<value_type>;

	template <typename otype> void copy(const dense<otype>&);
	void swap(dense&);
private:
	value_ptr _values;
public:
	value_type* values() const { return _values.data(); }

	template <typename otype>
	dense& operator=(const dense<otype>& o) { copy(o); return *this; }
	dense& operator=(const dense& o) { copy(o); return *this; }
	dense& operator=(dense&& o) { swap(o); return *this; }

	dense(size sz, value_ptr values) :
		base(sz), _values(std::move(values)) {}
	dense(int rows, int cols, value_ptr values) :
		dense({rows, cols}, std::move(values)) {}
	dense(size sz) : dense(sz, sz.rows * sz.cols) {}
	dense(int rows = 0, int cols = 0) : dense(size{rows, cols}) {}
	template <typename fill_fn>
	dense(size sz, const filler<fill_fn>& f) : dense(sz) { fill(*this, f); }
	template <typename fill_fn>
	dense(int rows, int cols, const filler<fill_fn>& f) :
		dense(size{rows, cols}, f) {}
	template <typename otype> explicit dense(const dense<otype>& o) :
		dense({0, 0}, nullptr) { copy(o); }
	dense(const dense& o) : dense({0, 0}, nullptr) { copy(o); }
	dense(dense&& o) : dense({0, 0}, nullptr) { swap(o); }
};

template <typename vtype>
template <typename otype>
void
dense<vtype>::
copy(const dense<otype>& o)
{
	const auto n = o.rows() * o.cols();
	value_ptr values(n, _values.get_allocator());

	auto* vdata = values.data();
	auto* wdata = o.values();
	auto k = [=] __device__ (int tid) { vdata[tid] = wdata[tid]; };
	util::transform<128, 8>(k, n);

	_values = move(values);
	base::operator=(o);
}

template <typename vtype>
void
dense<vtype>::
swap(dense& o)
{
	std::swap(_values, o._values);
	base::operator=(std::move(o));
}

template <typename vtype>
void
reshape(dense<vtype>& d, size sz)
{
	d = base{sz};
}

} // namespace linalg
