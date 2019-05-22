#pragma once
#include "util/launch.h"
#include "util/memory.h"
#include "base.h"
#include "types.h"

namespace linalg {

using std::move;

template <typename vtype>
class sparse : public base {
private:
	static_assert(is_field_v<vtype>, "type is not double, float, or complex");
public:
	using value_type = vtype;
	using index_type = typename base::index_type;
protected:
	using value_ptr = util::memory<value_type>;
	using index_ptr = util::memory<index_type>;

	template <typename otype> void copy(const sparse<otype>&);
	void swap(sparse&);
private:
	int _nonzero;
	index_ptr _indices;
	value_ptr _values;
public:
	int         nonzero() const { return _nonzero; }
	value_type* values()  const { return _values.data(); }
	index_type* indices() const { return _indices.data(); }

	template <typename otype>
	sparse& operator=(const sparse<otype>& o) { copy(o); return *this; }
	sparse& operator=(const sparse& o) { copy(o); return *this; }
	sparse& operator=(sparse&& o) { swap(o); return *this; }

	sparse(size sz, int nnz, index_ptr indices, value_ptr values) :
		base(sz), _nonzero(nnz), _indices(move(indices)), _values(move(values)) {}
	sparse(int rows, int cols, int nnz, index_ptr indices, value_ptr values) :
		sparse({rows, cols}, nnz, move(indices), move(values)) {}
	sparse(size sz, int nnz = 0) : sparse(sz, nnz, nnz, nnz) {}
	sparse(int rows = 0, int cols = 0, int nnz = 0) : sparse({rows, cols}, nnz) {}
	template <typename otype> explicit sparse(const sparse<otype>& o) :
		sparse({0, 0}, 0, nullptr, nullptr) { copy(o); }
	sparse(const sparse& o) : sparse({0, 0}, 0, nullptr, nullptr) { copy(o); }
	sparse(sparse&& o) : sparse({0, 0}, 0, nullptr, nullptr) { swap(o); }
};

template <typename vtype>
template <typename otype>
void
sparse<vtype>::
copy(const sparse<otype>& o)
{
	const auto n = o.nonzero();
	value_ptr values(n, _values.get_allocator());
	index_ptr indices(n, _indices.get_allocator());

	auto* vdata = values.data();
	auto* idata = indices.data();
	auto* wdata = o.values();
	auto* jdata = o.indices();
	auto k = [=] __device__ (int tid)
	{
		vdata[tid] = wdata[tid];
		idata[tid] = jdata[tid];
	};
	util::transform<128, 8>(k, n);

	_nonzero = n;
	_values = move(values);
	_indices = move(indices);
	base::operator=(o);
}

template <typename vtype>
void
sparse<vtype>::
swap(sparse& o)
{
	std::swap(_nonzero, o._nonzero);
	std::swap(_values, o._values);
	std::swap(_indices, o._indices);
	base::operator=(std::move(o));
}

} // namespace linalg
