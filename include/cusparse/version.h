#pragma once
#include <cusparse.h>
#include "handle.h"
#include "exceptions.h"

namespace cusparse {
	struct version {
		int major;
		int minor;

		bool operator==(const version& o) const { return major == o.major && minor == o.minor; }
		bool operator>(const version& o) const { return major > o.major || minor > o.minor; }
		bool operator!=(const version& o) const { return !(*this == o); }
		bool operator>=(const version& o) const { return *this == o || *this > o; }
		bool operator<(const version& o) const { return !(*this >= o); }
		bool operator<=(const version& o) const { return !(*this > o); }

		version(int combined) :
			major(combined / 1000),
			minor((combined % 1000) / 10) {}
	};

	inline std::ostream& operator<<(std::ostream& out, const version& v)
	{
		return out << v.major << '.' << v.minor;
	}

	inline version get_version(handle& h)
	{
		int v;
		throw_if_error(cusparseGetVersion(h, &v));
		return v;
	}
}
