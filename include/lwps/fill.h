#pragma once
#include "namespace.h"
#include "types.h"

namespace LWPS_NAMESPACE {
	class vector;

	namespace fill {
		class constant {
			private:
				value_type _value;
			public:
				constexpr constant(value_type value) :
					_value(value) {}
				__host__ __device__ operator value_type() const { return _value; }

			friend class LWPS_NAMESPACE::vector;
			friend constexpr constant operator+(const constant&, value_type);
			friend constexpr constant operator+(value_type, const constant&);
			friend constexpr constant operator-(const constant&, value_type);
			friend constexpr constant operator-(value_type, const constant&);
			friend constexpr constant operator*(const constant&, value_type);
			friend constexpr constant operator*(value_type, const constant&);
			friend constexpr constant operator/(const constant&, value_type);
		};

		static constexpr constant ones(1);
		static constexpr constant zeros(0);

		constexpr constant
		operator+(const constant& c, value_type v)
		{
			return constant(c._value + v);
		}

		constexpr constant
		operator+(value_type v, const constant& c)
		{
			return constant(c._value + v);
		}

		constexpr constant
		operator-(const constant& c, value_type v)
		{
			return constant(c._value - v);
		}

		constexpr constant
		operator-(value_type v, const constant& c)
		{
			return constant(c._value - v);
		}

		constexpr constant
		operator*(const constant& c, value_type v)
		{
			return constant(c._value * v);
		}

		constexpr constant
		operator*(value_type v, const constant& c)
		{
			return constant(c._value * v);
		}

		constexpr constant
		operator/(const constant& c, value_type v)
		{
			return constant(c._value / v);
		}
	}
}

using LWPS_NAMESPACE::fill::operator+;
using LWPS_NAMESPACE::fill::operator-;
using LWPS_NAMESPACE::fill::operator*;
using LWPS_NAMESPACE::fill::operator/;
