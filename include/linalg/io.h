#pragma once
#include <ostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include "cuda/copy.h"
#include "types.h"
#include "vector.h"
#include "matrix.h"

namespace linalg {
namespace io {

struct style {
	std::string begin_vector;
	std::string vector_delimiter;
	std::string end_vector;

	std::string begin_matrix;
	std::string matrix_delimiter;
	std::string end_matrix;

	std::string begin_row;
	std::string row_delimiter;
	std::string end_row;

	template <typename value_type>
	void
	generic_value(std::ostream& out, value_type v) const {
		if (real(v) >= 0) out << ' ';
		out << v;
	}

	virtual void prepare(std::ostream& out) const
	{
		//using limits = std::numeric_limits<double>;
		//static constexpr auto eps = limits::epsilon();
		//int prec = std::floor(-std::log10(eps));
		//out << std::fixed << std::setprecision(prec);
	}
	virtual void value(std::ostream& out, float v) const { generic_value(out, v); }
	virtual void value(std::ostream& out, double v) const { generic_value(out, v); }
	virtual void value(std::ostream& out, complex<double> v) const { generic_value(out, v); }
	virtual void value(std::ostream& out, complex<float> v) const { generic_value(out, v); }

	style(std::string bv, std::string vd, std::string ev,
	      std::string bm, std::string md, std::string em,
	      std::string br, std::string rd, std::string er) :
		begin_vector(bv), vector_delimiter(vd), end_vector(ev),
		begin_matrix(bm), matrix_delimiter(md), end_matrix(em),
		begin_row(br), row_delimiter(rd), end_row(er) {}
};

struct styler { std::ostream& out; const style& sty; };

namespace styles {

inline const
struct none_style : style {
	none_style() :
		style("", ", ", "", "", ", ", "", "", "\n ", "") {}
} none;

inline const
struct python_style : style {
	python_style() :
		style("[", ", ", "]", "[", ", ", "]", "[", ",\n ", "]") {}
} python;


inline const
struct numpy_style : style {
	numpy_style() :
		style("np.array([", ", ", "])", "np.array(\n[", ", ", "]\n)", "[", ",\n ", "]") {}
} numpy;

inline const
struct matlab_style : style {
	matlab_style() :
		style("[", ", ", "]'", "[", ", ", "]", "", ";\n ", "") {}
} matlab;

} // namespace styles

using namespace styles;

inline styler operator<<(std::ostream& out, const style& sty) { return styler{out, sty}; }

namespace detail {

struct setter { const style& sty; };

inline const style*&
get_default_style()
{
	static const style* ptr = &none;
	return ptr;
}

template <typename vtype>
inline int
scalexp(int n, const vtype* data)
{
	static constexpr int min_exp = std::numeric_limits<vtype>::min_exponent10;
	int exp = min_exp-1;
	for (int i = 0 ; i < n; ++i) {
		vtype datum = data[i];
		int curr = datum == 0 ? min_exp : std::ceil(std::log10(std::fabs(datum)/2));
		exp = curr > exp ? curr : exp;
	}
	return exp == min_exp-1 ? 0 : exp;
}

}

inline const style&
get_default_style()
{
	return *detail::get_default_style();
}

inline void
set_default_style(const style& sty)
{
	detail::get_default_style() = &sty;
}

inline auto setstyle(const style& sty) { return detail::setter{sty}; }

template <typename vtype>
std::ostream&
operator<<(styler styr, const vector<dense<vtype>>& v)
{
	auto& out = styr.out;
	auto& sty = styr.sty;
	sty.prepare(out);

	auto n = v.rows();
	auto* hdata = new vtype[n];
	cuda::dtoh(hdata, v.values(), n);

	int exp = detail::scalexp(n, hdata);
	base_t<vtype> scale = std::pow(10., -exp);
	if (exp) out << "1e" << exp << " * \\\n";
	out << std::fixed << std::setprecision(15);

	out << sty.begin_vector;
	for (int i = 0; i < n; ++i) {
		if (i) out << sty.vector_delimiter;
		sty.value(out, scale * hdata[i]);
	}
	delete[] hdata;
	return out << sty.end_vector;
}

template <typename vtype>
std::ostream&
operator<<(styler styr, const matrix<dense<vtype>>& m)
{
	auto& out = styr.out;
	auto& sty = styr.sty;
	sty.prepare(out);

	auto rows = m.rows();
	auto cols = m.cols();
	auto n = rows * cols;
	auto* hdata = new vtype[n];
	cuda::dtoh(hdata, m.values(), n);

	int exp = detail::scalexp(n, hdata);
	base_t<vtype> scale = std::pow(10., -exp);
	if (exp) out << "1e" << exp << " * ";

	out << sty.begin_matrix;
	for (int i = 0; i < rows; ++ i) {
		if (i) out << sty.row_delimiter;
		out << sty.begin_row;
		for (int j = 0; j < cols; ++j) {
			if (j) out << sty.matrix_delimiter;
			sty.value(out, scale * hdata[i + j * rows]);
		}
		out << sty.end_row;
	}
	delete[] hdata;
	return out << sty.end_matrix;
}

template <typename vtype>
std::ostream&
operator<<(styler styr, const vector<sparse<vtype>>& v)
{
	auto& out = styr.out;
	auto& sty = styr.sty;
	sty.prepare(out);

	auto rows = v.rows();
	auto nnz = v.nonzero();

	auto* h_indices = new int[nnz];
	auto* h_values = new vtype[nnz];
	cuda::dtoh(h_indices, v.indices(), nnz);
	cuda::dtoh(h_values, v.values(), nnz);

	int exp = detail::scalexp(nnz, h_values);
	base_t<vtype> scale = std::pow(10., -exp);
	if (exp) out << "1e" << exp << " * ";
	out << std::fixed << std::setprecision(15);

	out << sty.begin_vector;
	int offset = 0;
	for (int i = 0; i < rows; ++i) {
		if (i) out << sty.vector_delimiter;
		if (h_indices[offset] == i)
			sty.value(out, scale * h_values[offset++]);
		else
			sty.value(out, (vtype) 0);
	}
	delete[] h_indices;
	delete[] h_values;
	out << sty.end_vector;
	if (offset != nnz) out << " # vector continues?";
	return out;
}

template <typename vtype>
std::ostream&
operator<<(styler styr, const matrix<sparse<vtype>>& m)
{
	auto& out = styr.out;
	auto& sty = styr.sty;
	sty.prepare(out);

	auto rows = m.rows();
	auto cols = m.cols();
	auto nnz = m.nonzero();

	auto* h_starts = new int[rows + 1];
	auto* h_indices = new int[nnz];
	auto* h_values = new vtype[nnz];

	if (m.nonzero())
		cuda::dtoh(h_starts, m.starts(), rows + 1);
	else
		std::memset(h_starts, 0, sizeof(int) * (rows + 1));
	cuda::dtoh(h_indices, m.indices(), nnz);
	cuda::dtoh(h_values, m.values(), nnz);

	int exp = detail::scalexp(nnz, h_values);
	base_t<vtype> scale = std::pow(10., -exp);
	if (exp) out << "1e" << exp << " * \\\n";
	out << std::fixed << std::setprecision(4);

	out << sty.begin_matrix;
	for (int i = 0; i < rows; ++i) {
		auto& end = h_starts[i+1];
		auto offset = h_starts[i];
		if (i) out << sty.row_delimiter;
		out << sty.begin_row;
		for (int j = 0; j < cols; ++j) {
			if (j) out << sty.matrix_delimiter;
			if (offset < end && j == h_indices[offset])
				sty.value(out, scale * h_values[offset++]);
			else
				sty.value(out, (vtype) 0);
		}
		out << sty.end_row;
		if (offset != end) out << " # row continues?";
	}
	delete[] h_starts;
	delete[] h_indices;
	delete[] h_values;
	return out << sty.end_matrix;
}

inline std::ostream&
operator<<(std::ostream& out, detail::setter setter)
{
	set_default_style(setter.sty);
	return out;
}

template <template <typename> class container,
		 template <typename> class layout, typename vtype>
std::ostream&
operator<<(std::ostream& out, const container<layout<vtype>>& c)
{
	auto& style = get_default_style();
	return styler{out, style} << c;
}

} // namespace io
} // namespace linalg

using linalg::io::operator<<;
