#include <utility>
#include <iostream>
#include "lwps/vector.h"
#include "lwps/blas.h"
#include "util/launch.h"


namespace lwps {
	vector::vector() :
		matrix_base(0, 1),
		_values{nullptr} {}

	vector::vector(const matrix_base& base) :
		matrix_base(base),
		_values(rows()) {}

	vector::vector(index_type rows) :
		matrix_base(rows, 1),
		_values(rows) {}

	vector::vector(const matrix_base& base, value_ptr&& ptr) :
		matrix_base(base),
		_values(std::move(ptr))
	{ ptr = nullptr; }

	vector::vector(index_type rows, value_ptr&& ptr) :
		matrix_base(rows, 1),
		_values(std::move(ptr)) {}

	vector::vector(const matrix_base& base, const fill::constant& value) :
		matrix_base(base),
		_values(rows())
	{
		auto* vals = values();
		auto k = [=] __device__ (int tid) { vals[tid] = value; };
		util::transform<128, 11>(k, rows());
	}

	vector::vector(index_type rows, const fill::constant& value) :
		matrix_base(rows, 1),
		_values(rows)
	{
		auto* vals = values();
		auto k = [=] __device__ (int tid) { vals[tid] = value; };
		util::transform<128, 11>(k, rows);
	}

	/*vector::vector(std::initializer_list<value_type> values) :
		matrix_base(values.size(), 1),
		_values{values} {}*/

	vector::vector(const vector& other) :
		matrix_base(other),
		_values(other.rows())
	{ copy(other); }

	vector::vector(vector&& other) :
		matrix_base((vector&&) other),
		_values(nullptr)
	{ swap(other); }

	void
	vector::copy(const vector& other)
	{
		auto rows = other.rows();
		auto* dval = values();
		auto* sval = other.values();

		auto k = [=] __device__ (int tid) { dval[tid] = sval[tid]; };
		util::transform<128, 11>(k, rows);
	}

	void
	vector::swap(vector& other)
	{
		std::swap(_values, other._values);
	}

	vector&
	vector::operator=(const vector& other)
	{
		return *this = vector{other};
	}

	vector&
	vector::operator=(vector&& other)
	{
		swap(other);
		matrix_base::operator=(std::move(other));
		return *this;
	}

	vector&
	vector::operator+=(const vector& other)
	{
		axpy(1.0, other, *this);
		return *this;
	}

	vector&
	vector::operator-=(const vector& other)
	{
		axpy(-1.0, other, *this);
		return *this;
	}

	vector&
	vector::operator*=(value_type k)
	{
		scal(k, *this);
		return *this;
	}

	vector&
	vector::operator/=(value_type k)
	{
		scal(1.0 / k, *this);
		return *this;
	}

	vector&
	vector::operator%=(const vector& other)
	{
		(void) (size(*this) % size(other));
		auto* lvals = values();
		auto* rvals = other.values();

		auto k = [=] __device__ (int tid) { lvals[tid] *= rvals[tid]; };
		util::transform<128, 11>(k, rows());
		return *this;
	}
}
