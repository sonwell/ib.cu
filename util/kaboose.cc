#include "util/functional.h"
#include "bases/types.h"

struct state {
	bases::vector u[3];
	bases::vector ub[3];
	bases::matrix x;
};

std::istream&
operator>>(std::istream& in, state& st)
{
	using namespace util::functional;
	auto load = [&] (auto& v) { in >> linalg::io::binary >> v; };
	map(load, st.u);
	map(load, st.ub);
	load(st.x);
	return in;
}

template <typename callback_type>
bool
read_state(std::istream& in, callback_type&& callback)
{
	state t;
	in >> t;
	if (in.eof()) return false;
	callback(std::move(t));
	return true;
}

std::ostream&
operator<<(std::ostream& out, const state& st)
{
	using namespace util::functional;
	auto store = [&] (const auto& v) { out << linalg::io::binary << v; };
	map(store, st.u);
	map(store, st.ub);
	store(st.x);
	return out;
}

int
main(void)
{
	util::set_default_resource(cuda::default_device().memory());

	state t;
	auto store = [&] (state s) { t = std::move(s); };
	if (!read_state(std::cin, store))
		return -1;

	while (read_state(std::cin, store));
	std::cout << t;
	return 0;
}
