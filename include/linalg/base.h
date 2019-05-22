#pragma once
#include "size.h"

namespace linalg {

struct base {
private:
	size _sz;
public:
	using index_type = int;

	constexpr int rows() const { return _sz.rows; }
	constexpr int cols() const { return _sz.cols; }

	base& operator=(const base&) = default;
	base& operator=(base&&) = default;
	constexpr explicit operator size() const { return _sz; }

	constexpr base(const base& b) : _sz(b._sz) {}
	constexpr base(size sz) : _sz(sz) {}
	constexpr base(int rows, int cols) :
		base(size{rows, cols}) {}
};

}
