#pragma once
#include "base.h"
#include "fill.h"
#include "io.h"
#include "device_ptr.h"
#include "cuda/copy.h"

namespace lwps {
	class vector : public matrix_base {
		private:
			using value_ptr = mem::device_ptr<value_type>;
		protected:
			value_ptr _values;
		public:
			vector();
			vector(const matrix_base&);
			vector(index_type);
			vector(const matrix_base&, value_ptr&&);
			vector(index_type, value_ptr&&);
			vector(index_type, const fill::constant&);
			vector(const matrix_size& s, const fill::constant& c) :
				vector(s.rows, c) {}
			vector(const matrix_base&, const fill::constant&);
			//vector(std::initializer_list<value_type>);
			vector(const vector&);
			vector(vector&&);
			vector(const matrix_size& size) : vector(size.rows) {}

			void copy(const vector&);
			void swap(vector&);

			const value_type* values() const { return _values; }
			value_type* values() { return _values; }
			vector& operator=(const vector&);
			vector& operator=(vector&&);
			vector& operator+=(const vector&);
			vector& operator-=(const vector&);
			vector& operator*=(value_type);
			vector& operator/=(value_type);
			vector& operator%=(const vector&);
	};

	inline vector
	operator+(vector left, const vector& right)
	{ return std::move(left += right); }

	inline vector
	operator-(vector left, const vector& right)
	{ return std::move(left -= right); }

	inline vector
	operator*(value_type a, vector m)
	{ return std::move(m *= a); }

	inline vector
	operator*(vector m, value_type a)
	{ return std::move(m *= a); }

	inline vector
	operator/(value_type a, vector m)
	{ return std::move(m /= a); }

	inline vector
	operator/(vector m, value_type a)
	{ return std::move(m /= a); }

	inline vector
	operator%(vector left, const vector& right)
	{ return std::move(left %= right); }

	inline vector
	operator-(vector left)
	{ return std::move(left *= -1); }

	inline vector
	operator+(vector left)
	{ return std::move(left); }

	inline std::ostream&
	operator<<(std::ostream& out, const vector& v)
	{
		auto* style = lwps::io::get_style();
		double* h_values = new double[v.rows()];
		cuda::dtoh(h_values, v.values(), v.rows());
		style->prepare(out);
		out << style->begin_vector;
		for (int i = 0; i < v.rows(); ++i) {
			if (i) out << style->vector_entry_delimiter;
			style->value(out, h_values[i]);
		}
		out << style->end_vector;
		delete[] h_values;
		return out;
	}
}
