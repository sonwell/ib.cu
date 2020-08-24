#pragma once
#include "util/functional.h"
#include "linalg/io.h"
#include "types.h"
#include "units.h"

namespace ins {

template <std::size_t dimensions>
struct state {
	static constexpr auto to_array = [] (auto&& v)
	{
		auto k = [] (auto&& ... vs) { return std::array{std::forward<decltype(vs)>(vs)...}; };
		return apply(k, std::forward<decltype(v)>(v));
	};

	using velocity = std::array<vector, dimensions>;
	using pressure = vector;

	units::time t;
	velocity u;
	pressure p;

	state() = default;

	state(units::time t, velocity u, pressure p) :
		t(t), u(std::move(u)), p(std::move(p)) {}

	template <typename u_type>
	state(units::time t, u_type&& u, pressure p) :
		state(t, to_array(std::forward<u_type>(u)), std::move(p)) {}
};

template <typename u_type>
state(units::time t, u_type&& u, vector p) ->
	state<std::tuple_size_v<std::decay_t<u_type>>>;

template <typename u_type, std::size_t dimensions>
state(units::time t, const u_type (&u)[dimensions], vector p) ->
	state<dimensions>;

} // namespace ins

namespace linalg {
namespace io {
namespace formatting {

template <typename format_type, std::size_t dimensions>
decltype(auto)
operator<<(writer<format_type> writer, const ins::state<dimensions>& state)
{
	writer << state.t;
	for (const ib::vector& u: state.u)
		writer << u;
	writer << state.p;
	return writer;
}

template <typename format_type, std::size_t dimensions>
decltype(auto)
operator>>(reader<format_type> reader, ins::state<dimensions>& state)
{
	reader >> state.t;
	for (ib::vector& u: state.u)
		reader >> u;
	reader >> state.p;
	return reader;
}

} // namespace formatting
} // namespace io
} // namespace linalg
