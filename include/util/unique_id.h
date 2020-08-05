#pragma once

namespace util {

using unique_id_t = const int *;

namespace impl {

template <int n>
struct flag {
	friend constexpr bool adl_flag(flag);
};

template <int n>
struct writer {
	static constexpr auto value = n;
	friend constexpr bool adl_flag(flag<n>) { return true; }
};

template <int n>
static constexpr unique_id_t
reader(float, flag<n>)
{
	// Always true, just instantiates the next result
	static_assert(writer<n>::value >= 0);
	return &writer<n>::value;
}

template <int n = 0,
		  bool = adl_flag(flag<n>{}),
		  unique_id_t r = reader(0, flag<n+1>{})>
static constexpr unique_id_t
reader(int, flag<n>, unique_id_t s = r)
{
	return s;
}

}

template <int n = 0,
          unique_id_t r = impl::reader(0, impl::flag<n>{})>
constexpr unique_id_t
unique_id(unique_id_t s = r)
{
	return s;
}

}
