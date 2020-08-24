#pragma once
#include <ostream>
#include <cassert>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <limits>
#include <array>
#include <tuple>
#include "util/getset.h"
#include "cuda/copy.h"
#include "cuda/device.h"
#include "types.h"
#include "vector.h"
#include "matrix.h"

namespace linalg {
namespace io {
namespace formatting {

struct binary;
struct text;
struct algebraic;
struct format;

namespace detail {

template <typename object_type>
inline constexpr auto is_linalg_v = std::is_base_of_v<base, object_type>;

}

struct format {};

template <typename format_type>
struct writer {
	static_assert(std::is_base_of_v<format, format_type>,
			"formats must inherit from formatting::format");
	std::ostream& stream;
	const format_type& format;

	writer(std::ostream& stream, const format_type& format) :
		stream(stream), format(format) {}

	template <typename other_format_type,
	          typename = std::enable_if_t<std::is_base_of_v<format_type, other_format_type>>>
	writer(const writer<other_format_type>& writer) :
		stream(writer.stream), format(writer.format) {}
};

template <typename format_type>
struct reader {
	static_assert(std::is_base_of_v<format, format_type>,
			"formats must inherit from formatting::format");
	std::istream& stream;
	const format_type& format;

	reader(std::istream& stream, const format_type& format) :
		stream(stream), format(format) {}

	template <typename other_format_type,
	          typename = std::enable_if_t<std::is_base_of_v<format_type, other_format_type>>>
	reader(const reader<other_format_type>& reader) :
		stream(reader.stream), format(reader.format) {}
};

template <template <typename> typename> struct container_magic;
template <> struct container_magic<linalg::vector> :
	std::integral_constant<int, 0x65760000> {}; // ve (vector)
template <> struct container_magic<linalg::matrix> :
	std::integral_constant<int, 0x616d0000> {}; // ma (matrix)

template <template <typename> typename> struct layout_magic;
template <> struct layout_magic<linalg::dense> :
	std::integral_constant<int, 0x00006400> {}; // d (dense)
template <> struct layout_magic<linalg::sparse> :
	std::integral_constant<int, 0x00007300> {}; // s (sparse)

template <typename> struct type_magic;
template <> struct type_magic<float> :
	std::integral_constant<int, 0x00000073> {}; // s (single precision)
template <> struct type_magic<double> :
	std::integral_constant<int, 0x00000064> {}; // d (double precision)

template <typename> struct magic_number;
template <template <typename> typename container,
          template <typename> typename layout,
          typename vtype>
struct magic_number<container<layout<vtype>>> {
	static constexpr int value =
		container_magic<container>::value |
		layout_magic<layout>::value       |
		type_magic<vtype>::value;
};

template <typename T>
inline constexpr auto magic_number_v = magic_number<T>::value;

inline struct binary : format {} binary;

inline struct text : format {
	std::string comment = "/!\\";

	// vector formatting
	std::string vector_begin = "";
	std::string vector_delim = "\n";
	std::string vector_end = "";

	// matrix formatting
	std::string matrix_begin = "";
	std::string row_begin = "";
	std::string matrix_delim = ", ";
	std::string row_end = "";
	std::string row_delim = "\n";
	std::string matrix_end = "";

	text(std::string comment,      std::string vector_begin,
	     std::string vector_delim, std::string vector_end,
	     std::string matrix_begin, std::string row_begin,
	     std::string matrix_delim, std::string row_end,
	     std::string row_delim,    std::string matrix_end) :
		comment(comment),           vector_begin(vector_begin),
		vector_delim(vector_delim), vector_end(vector_end),
		matrix_begin(matrix_begin), row_begin(row_begin),
		matrix_delim(matrix_delim), row_end(row_end),
		row_delim(row_delim),       matrix_end(matrix_end) {}
	text() {}
} none;

struct algebraic : text {
	std::string continuation = "↩";

	algebraic(std::string comment,      std::string vector_begin,
	          std::string vector_delim, std::string vector_end,
	          std::string matrix_begin, std::string row_begin,
	          std::string matrix_delim, std::string row_end,
	          std::string row_delim,    std::string matrix_end,
		      std::string continuation) :
		text{comment,      vector_begin, vector_delim, vector_end,
		     matrix_begin, row_begin,    matrix_delim, row_end,
		     row_delim,    matrix_end},
		continuation(continuation) {}
	algebraic() {}
};

inline struct python : text {
	python(std::string padding = "") :
		text{" #", "[\n" + padding + " ",
			",\n" + padding + " ", "\n]",
			"[\n" + padding + "    ",
			"[", ", ", "]",
			",\n" + padding + "    ", "]"} {}
} python;

inline struct numpy : algebraic {
private:
	struct private_tag {};
	numpy(private_tag, std::string imported_name, std::string padding) :
		algebraic{" #",
			imported_name + ".array([\n" + padding, ",\n" + padding, "\n])",
			imported_name + ".array([\n" + padding, "[", ", ", "]", ",\n" + padding, "])", "\\"} {}
public:
	numpy(std::string imported_name = "np", std::string padding = "") :
		numpy(private_tag{}, imported_name, padding + "    ") {}
} numpy;

inline struct matlab : algebraic {
	matlab(std::string padding = "") :
		algebraic{"%", "[", "\n" + padding + " ", "]",
		"[", "", ", ", "", "\n" + padding + " ", "]", "..."} {}
} matlab;

namespace detail {

template <typename vtype>
int
zeros(vtype&& v)
{
	return std::ceil(std::log10(std::abs(v)/2));
}

template <typename vtype>
int
exponent(int n, vtype* values)
{
	using limits = std::numeric_limits<vtype>;
	static constexpr int min_exp = limits::min_exponent10;
	int exp = min_exp - 1;
	for (int i = 0; i < n; ++i) {
		auto& datum = values[i];
		int curr = !datum ? min_exp-1 : zeros(datum);
		exp = curr > exp ? curr : exp;
	}
	return exp < min_exp ? 0 : exp;
}

template <typename value_type>
void
bytes(std::ostream& out, const value_type* v, std::size_t n)
{
	out.write(reinterpret_cast<const char*>(v), n * sizeof(value_type));
}

template <typename value_type>
void
bytes(std::istream& in, value_type* v, std::size_t n)
{
	if (in.eof()) return;
	in.read(reinterpret_cast<char*>(v), n * sizeof(value_type));
}

inline std::array<int, 3>
header(std::istream& in)
{
	std::array<int, 3> buf;
	bytes(in, buf.data(), 3);
	return buf;
}

template <typename value_type>
struct tmpflags {
	std::ostream& out;
	std::ios::fmtflags flags;
	std::streamsize precision;
	std::streamsize width;

	tmpflags(const tmpflags&) = delete;
	tmpflags(tmpflags&&) = delete;
	tmpflags(std::ostream& out) :
		out(out),
		flags(out.flags()),
		precision(out.precision()),
		width(out.width())
	{
		using btype = base_t<value_type>;
		using limits = std::numeric_limits<btype>;
		int prec = -zeros(limits::epsilon());
		out << std::fixed << std::setprecision(prec);
	}

	~tmpflags()
	{
		out.flags(flags);
		out.precision(precision);
		out.width(width);
	}
};

template <typename value_type>
void
text_value(std::ostream& out, value_type v)
{
	if (1.0 / real(v) > 0) out << ' ';
	out << v;
}

template <typename value_type>
void
write(std::ostream& out, const text& fmt, base_t<value_type> scale,
		const vector<dense<value_type>>& v)
{
	tmpflags<value_type> flags{out};
	auto* w = v.values();
	int n = v.rows();

	out << fmt.vector_begin;
	for (int i = 0; i < n; ++i) {
		if (i) out << fmt.vector_delim;
		text_value(out, w[i] * scale);
	}
	out << fmt.vector_end;
}

template <typename value_type>
void
write(std::ostream& out, const text& fmt, base_t<value_type> scale,
		const vector<sparse<value_type>>& v)
{
	tmpflags<value_type> flags{out};
	auto* w = v.values();
	int* k = v.indices();
	int n = v.rows();
	int nnz = v.nonzero();
	int offset = 0;

	out << fmt.vector_begin;
	for (int i = 0; i < n; ++i) {
		if (i) out << fmt.vector_delim;
		if (offset < nnz && k[offset] == i)
			text_value(out, w[offset++] * scale);
		else
			text_value(out, (value_type) 0);
	}
	out << fmt.vector_end;
	if (offset < nnz)
		out << ' ' << fmt.comment << " vector continues?";
}

template <typename value_type>
void
write(std::ostream& out, const text& fmt, base_t<value_type> scale,
		const matrix<dense<value_type>>& v)
{
	tmpflags<value_type> flags{out};
	auto* w = v.values();
	int n = v.rows();
	int m = v.cols();

	out << fmt.matrix_begin;
	for (int i = 0; i < n; ++i) {
		if (i) out << fmt.row_delim;
		out << fmt.row_begin;
		for (int j = 0; j < m; ++j) {
			if (j) out << fmt.matrix_delim;
			text_value(out, w[i + j * n] * scale);
		}
		out << fmt.row_end;
	}
	out << fmt.matrix_end;
}

template <typename value_type>
void
write(std::ostream& out, const text& fmt, base_t<value_type> scale,
		const matrix<sparse<value_type>>& v)
{
	tmpflags<value_type> flags{out};
	auto* w = v.values();
	auto* s = v.starts();
	auto* k = v.indices();
	int n = v.rows();
	int m = v.cols();
	int nnz = v.nonzero();

	out << fmt.matrix_begin;
	for (int i = 0; i < n; ++i) {
		if (i) out << fmt.row_delim;
		out << fmt.row_begin;
		auto end = nnz ? s[i+1] : 0;
		auto offset = nnz ? s[i] : 0;
		for (int j = 0; j < m; ++j) {
			if (j) out << fmt.matrix_delim;
			if (offset < end && k[offset] == j)
				text_value(out, w[offset++] * scale);
			else
				text_value(out, (value_type) 0);
		}
		out << fmt.row_end;
		if (offset < end)
			out << ' ' << fmt.comment << " row continues ?";
	}
	out << fmt.matrix_end;
}

template <typename object_type>
void
write(std::ostream& out, const format&, const object_type& v)
{
	out << v;
}

template <typename object_type,
		typename = std::enable_if_t<is_linalg_v<object_type>>>
void
magic(std::ostream& out, const object_type& v)
{
	static constexpr int magic_number = magic_number_v<object_type>;
	bytes(out, &magic_number, 1);
}

template <typename object_type,
		typename = std::enable_if_t<is_linalg_v<object_type>>>
void
magic(std::istream& in, const object_type& v)
{
	static constexpr int magic_number = magic_number_v<object_type>;
	int n;
	bytes(in, &n, 1);
	assert(n == magic_number);
}

template <typename value_type>
auto
nonzeros(const sparse<value_type>& v)
{
	return v.nonzero();
}

template <typename value_type>
auto
nonzeros(const dense<value_type>& v)
{
	return v.rows() * v.cols();
}

template <typename object_type,
		typename = std::enable_if_t<detail::is_linalg_v<object_type>>>
void
write(std::ostream& out, const text& fmt, const object_type& v)
{
	detail::write(out, fmt, 1.0, v);
}

template <typename object_type,
		typename = std::enable_if_t<detail::is_linalg_v<object_type>>>
void
write(std::ostream& out, const algebraic& fmt, const object_type& v)
{
	int exp = detail::exponent(nonzeros(v), v.values());
	if (exp) out << "1e" << exp << " * " << fmt.continuation << '\n';
	write(out, fmt, std::pow(10., -exp), v);
}

template <typename object_type>
void
write(std::ostream& out, const struct binary&, const object_type& object)
{
	bytes(out, &object, 1);
}

template <typename value_type>
void
write(std::ostream& out, const struct binary&, const vector<dense<value_type>>& v)
{
	int data[3] = {v.rows(), 1, v.rows()};
	auto* d = v.values();
	magic(out, v);
	bytes(out, data, 3);
	bytes(out, d, data[2]);
}

template <typename value_type>
void
write(std::ostream& out, const struct binary&, const vector<sparse<value_type>>& v)
{
	int data[3] = {v.rows(), 1, v.nonzero()};
	int* k = v.indices();
	auto* d = v.values();
	magic(out, v);
	bytes(out, data, 3);
	bytes(out, k, data[2]);
	bytes(out, d, data[2]);
}

template <typename value_type>
void
write(std::ostream& out, const struct binary&, const matrix<dense<value_type>>& v)
{
	int data[3] = {v.rows(), v.cols(), v.rows() * v.cols()};
	auto* d = v.values();
	magic(out, v);
	bytes(out, data, 3);
	bytes(out, d, data[2]);
}

template <typename value_type>
void
write(std::ostream& out, const struct binary&, const matrix<sparse<value_type>>& v)
{
	int data[3] = {v.rows(), v.cols(), v.nonzero()};
	int* s = v.starts();
	int* k = v.indices();
	auto* d = v.values();
	magic(out, v);
	bytes(out, data, 3);
	bytes(out, s, data[0] ? data[0] + 1 : 0);
	bytes(out, k, data[2]);
	bytes(out, d, data[2]);
}

template <typename object_type>
void
read(std::istream& in, const format&, object_type& v)
{
	in >> v;
}

template <typename object_type>
void
read(std::istream& in, const struct binary&, object_type& object)
{
	bytes(in, &object, 1);
}

template <typename value_type>
void
read(std::istream& in, const struct binary& fmt, vector<dense<value_type>>& v)
{
	magic(in, v);
	auto&& [n, m, nnz] = header(in);
	util::memory<value_type> buf(n, util::new_delete_resource());
	bytes(in, buf.data(), n);
	v = {n, std::move(buf)};
}

template <typename value_type>
void
read(std::istream& in, const struct binary& fmt, vector<sparse<value_type>>& v)
{
	magic(in, v);
	auto&& [n, m, nnz] = header(in);
	util::memory<value_type> vals(nnz, util::new_delete_resource());
	util::memory<int>        inds(nnz, util::new_delete_resource());
	bytes(in, inds.data(), nnz);
	bytes(in, vals.data(), nnz);
	v = {n, nnz, std::move(inds), std::move(vals)};
}

template <typename value_type>
void
read(std::istream& in, const struct binary& fmt, matrix<dense<value_type>>& v)
{
	magic(in, v);
	auto&& [n, m, nnz] = header(in);
	util::memory<value_type> buf(n * m, util::new_delete_resource());
	bytes(in, buf.data(), n * m);
	v = {n, m, std::move(buf)};
}

template <typename value_type>
void
read(std::istream& in, const struct binary& fmt, matrix<sparse<value_type>>& v)
{
	magic(in, v);
	auto&& [n, m, nnz] = header(in);
	util::memory<value_type> vals(nnz, util::new_delete_resource());
	util::memory<int>        inds(nnz, util::new_delete_resource());
	util::memory<int>        rows(n ? n+1 : 0, util::new_delete_resource());
	bytes(in, rows.data(), n ? n+1 : 0);
	bytes(in, inds.data(), nnz);
	bytes(in, vals.data(), nnz);
	v = {n, m, nnz, std::move(rows), std::move(inds), std::move(vals)};
}

template <typename value_type, typename transfer_type>
auto
copy(const vector<dense<value_type>>& v, transfer_type&& transfer)
	-> std::decay_t<decltype(v)>
{
	auto n = v.rows();
	auto* w = v.values();
	return {n, transfer(n, w)};
}

template <typename value_type, typename transfer_type>
auto
copy(const vector<sparse<value_type>>& v, transfer_type&& transfer)
	-> std::decay_t<decltype(v)>
{
	auto n = v.rows();
	auto nnz = v.nonzero();
	auto* w = v.values();
	auto* k = v.indices();
	return {n, nnz, transfer(nnz, w), transfer(nnz, k)};
}

template <typename value_type, typename transfer_type>
auto
copy(const matrix<dense<value_type>>& v, transfer_type&& transfer)
	-> std::decay_t<decltype(v)>
{
	auto n = v.rows();
	auto m = v.cols();
	auto* w = v.values();
	return {n, m, transfer(n * m, w)};
}

template <typename value_type, typename transfer_type>
auto
copy(const matrix<sparse<value_type>>& v, transfer_type&& transfer)
	-> std::decay_t<decltype(v)>
{
	auto n = v.rows();
	auto m = v.cols();
	auto nnz = v.nonzero();
	auto* s = v.starts();
	auto* k = v.indices();
	auto* w = v.values();
	return {n, m, nnz, transfer(nnz ? n+1 : 0, s),
		transfer(nnz, k), transfer(nnz, w)};
}

template <typename wrapped_type, typename transfer_type>
decltype(auto)
copy(const util::getset<wrapped_type>& gs, transfer_type&& transfer)
{
	return copy((const wrapped_type&) gs, std::forward<transfer_type>(transfer));
}

template <typename object_type, typename transfer_type>
decltype(auto)
copy(const object_type& object, transfer_type&&)
{
	return object;
}

} // namespace detail

template <typename format_type, typename object_type>
void
read(std::istream& in, const format_type& fmt, object_type& v)
{
	auto copy = [] (int n, const auto* v)
	{
		using value_type = std::decay_t<decltype(*v)>;
		using buffer = util::memory<value_type>;
		buffer buf(n, cuda::get_device().memory());
		cuda::htod(buf.data(), v, n);
		return buf;
	};

	if (in.peek() == EOF) return;
	detail::read(in, fmt, v);
	v = detail::copy(v, copy);
}

template <typename format_type, typename object_type>
void
write(std::ostream& out, const format_type& fmt, const object_type& v)
{
	auto copy = [] (int n, const auto* v)
	{
		using value_type = std::decay_t<decltype(*v)>;
		using buffer = util::memory<value_type>;
		buffer buf(n, util::new_delete_resource());
		cuda::dtoh(buf.data(), v, n);
		return buf;
	};

	detail::write(out, fmt, detail::copy(v, copy));
}

template <typename format_type, typename object_type>
decltype(auto)
operator<<(writer<format_type> wr, const object_type& object)
{
	std::ostream& stream = wr.stream;
	const auto& fmt = wr.format;
	auto cb = [&] (const auto& fmt) { write(stream, fmt, object); };

	if constexpr (std::is_base_of_v<format, std::decay_t<object_type>>)
		return writer{stream, object};
	else { cb(fmt); return wr; }
}

template <typename format_type, typename object_type>
decltype(auto)
operator>>(reader<format_type> rd, object_type& object)
{
	std::istream& stream = rd.stream;
	const auto& fmt = rd.format;
	auto cb = [&] (const auto& fmt) { read(stream, fmt, object); };

	if constexpr (std::is_base_of_v<format, std::decay_t<object_type>>)
		return reader{stream, object};
	else { cb(fmt); return rd; }
}

template <typename format_type>
decltype(auto)
operator<<(writer<format_type> wr, std::ostream& (*func)(std::ostream&))
{
	return wr.stream << *func;
}

template <typename format_type,
          typename = std::enable_if_t<std::is_base_of_v<format, format_type>>>
decltype(auto)
operator<<(std::ostream& out, const format_type& fmt)
{
	return writer{out, fmt};
}

template <typename format_type,
          typename = std::enable_if_t<std::is_base_of_v<format, format_type>>>
decltype(auto)
operator>>(std::istream& out, const format_type& fmt)
{
	return reader{out, fmt};
}

} // namespace formatting

using formatting::binary;
using formatting::numpy;
using formatting::python;
using formatting::matlab;

} // namespace io
} // namespace linalg
