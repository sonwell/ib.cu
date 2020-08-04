#pragma once
#include <ostream>

namespace util {

struct version {
	int major;
	int minor;
	int patch;

	constexpr version(int major, int minor, int patch = 0) :
		major(major), minor(minor), patch(patch) {}
};

constexpr bool
operator==(const version& l, const version& r)
{
	return l.major == r.major &&
		l.minor == r.minor &&
		l.patch == r.patch;
}

constexpr bool
operator>(const version& l, const version& r)
{
	return (l.major > r.major) || ((l.major == r.major &&
	        l.minor > r.minor) || ((l.minor == r.minor &&
	        l.patch > r.patch)));
}

constexpr bool
operator>=(const version& l, const version& r)
{
	return l > r || l == r;
}

constexpr bool
operator!=(const version& l, const version& r)
{
	return !(l == r);
}

constexpr bool
operator<(const version& l, const version& r)
{
	return !(l >= r);
}

constexpr bool
operator<=(const version& l, const version& r)
{
	return !(l > r);
}

inline std::ostream&
operator<<(std::ostream& out, const version& v)
{
	return out << v.major << '.' << v.minor << '.' << v.patch;
}

}
