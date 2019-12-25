#include "bases/types.h"

struct state {
	bases::vector u, v, w;
	bases::matrix x;
};

std::istream&
operator>>(std::istream& in, state& st)
{
	auto load = [&] (auto& v) { in >> linalg::io::binary >> v; };
	load(st.u);
	load(st.v);
	load(st.w);
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
	auto store = [&] (const auto& v) { out << linalg::io::binary << v; };
	store(st.u);
	store(st.v);
	store(st.w);
	store(st.x);
	return out;
}

int
main(void)
{
	util::set_default_resource(cuda::default_device().memory());

	state t;
	auto store = [&] (state&& s) { t = std::move(s); };
	if (!read_state(std::cin, store))
		return -1;

	while (read_state(std::cin, store));
	std::cout << t;
	return 0;
}
