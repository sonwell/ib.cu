#include "util/functional.h"
#include "bases/types.h"
#include "units.h"

struct state {
	units::time t;
	bases::vector u[3];
	bases::vector p;
	bases::matrix r;
	bases::matrix s;
	bases::matrix e;
};

std::istream&
operator>>(std::istream& in, state& st)
{
	using namespace util::functional;
	auto load = [&] (auto& v) { in >> linalg::io::binary >> v; };
	load(st.t);
	map(load, st.u);
	load(st.p);
	load(st.r);
	load(st.s);
	load(st.e);
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
	store(st.t);
	map(store, st.u);
	store(st.p);
	store(st.r);
	store(st.s);
	store(st.e);
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
