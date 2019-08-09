#include <iostream>
#include "util/counter.h"

template <int>
struct counter {
	static constexpr util::counter value = 0u;
};

template <int n>
inline constexpr auto counter_v = counter<n>::value;

struct tester {
	unsigned id;

	template <int n=0, unsigned m = std::next(counter_v<n>)>
	constexpr tester() :
		id(m) {}
};

int
main(void)
{
	tester a, b;

	std::cout << a.id << ' ' << b.id << std::endl;

	return 0;
}
