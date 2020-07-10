#pragma once
#include <ostream>
#include <stdexcept>

namespace util {

// An adaptor for enums that have the same meaning but possibly different values
// from different libraries. Uses `size_t` as an intermediary and can be cast to
// any of the enum types. E.g.,
//
//
//     // library1_adaptor.hpp
//     using enum1_container = enum_container<enum1, value1, value2, value3>;
//     // etc.
//
//     // library2_adaptor.hpp
//     using enum2_container = enum_container<enum2, foo, bar, baz>;
//     // etc.
//
//     // my_library.hpp
//     enum modes_enum { chevre, ementaler, gouda, cheddar };
//     using mode_container = enum_container<modes_enum, gouda, cheddar, ementaler>;
//     // value1 means foo means gouda, value2 means bar means cheddar, etc
//     using mode = adaptor<mode_container, enum1_container, enum2_container>;
//
//     inline void init(context& ctx, mode m)
//     {
//         auto dispatch = overload{
//             // casts m to enum1 for call to library 1
//             [&] (context1& ctx) { init1(ctx, m); },
//             // casts m to enum2 for call to library 2
//             [&] (context2& ctx) { init2(ctx, m); }
//         };
//         visit(ctx, dispatch); // dispatch based on polymorphic type of ctx
//     }
//
//     I use this for e.g. LAPACK/BLAS operations (i.e., conjugate transpose, etc.)
//     enums from cuBLAS, cuSPARSE, cuSOLVER, which all mean the same thing but
//     are different enum types.

// Since we are potentially only listing some valid enum values, an enum value
// may be passed to the adaptor constructor that does not match with any of the
// values listed. In that case, throw an error.
struct bad_enum_value : public std::runtime_error {
	bad_enum_value() : std::runtime_error("bad enum value") {}
	bad_enum_value(const char* what_arg) :
		std::runtime_error(what_arg) {}
	bad_enum_value(std::string& what_arg) :
		std::runtime_error(what_arg) {}
};

template <typename enum_type, enum_type ... enum_values>
struct enum_container {
	static_assert(std::is_enum<enum_type>::value,
			"enum_container only allows enum types");
	using value_type = enum_type;
	static constexpr enum_type values[] = {enum_values...};
	static constexpr auto size() { return sizeof...(enum_values); }
protected:
	static constexpr auto
	get_index(value_type value)
	{
		for (std::size_t i = 0; i < size(); ++i)
			if (values[i] == value)
				return i;
		throw bad_enum_value();
	}

	static constexpr auto
	get_value(value_type& value, std::size_t index)
	{
		value = values[index];
	}

	constexpr enum_container() = default;
};

namespace detail {

template <typename ... types> struct unique;

template <> struct unique<> : std::true_type {};
template <typename type> struct unique<type> : std::true_type {};

template <typename first, typename ... rest>
struct unique<first, rest...> :
	std::integral_constant<bool, unique<rest...>::value &&
		!(std::is_same_v<first, rest> || ...)> {};

} // namespace detail

template <typename container, typename ... containers>
class adaptor : protected container, protected containers ... {
private:
	template <typename T> using is_adapted =
		std::integral_constant<bool,
			std::is_same_v<typename container::value_type, T> ||
			(std::is_same_v<typename containers::value_type, T> || ...)>;
	static_assert(((container::size() == containers::size()) && ...),
			"enums must have the same number of elements");
	static_assert(detail::unique<typename container::value_type,
	                     typename containers::value_type...>::value,
	              "enum types must be unique");
	using container::get_index;
	using container::get_value;
	using containers::get_index...;
	using containers::get_value...;

	std::size_t index;
public:
	using container::size;

	template <typename enum_type, typename = std::enable_if_t<is_adapted<enum_type>::value>>
	constexpr operator enum_type() const { enum_type v; get_value(v, index); return v; }

	constexpr bool operator==(const adaptor& o) const { return o.index == index; }
	constexpr bool operator!=(const adaptor& o) const { return o.index != index; }
	template <typename enum_type, typename = std::enable_if_t<is_adapted<enum_type>::value>>
	constexpr bool operator==(enum_type o) const { return get_index(o) == index; }
	template <typename enum_type, typename = std::enable_if_t<is_adapted<enum_type>::value>>
	constexpr bool operator!=(enum_type o) const { return get_index(o) != index; }

	template <typename enum_type, typename = std::enable_if_t<is_adapted<enum_type>::value>>
	constexpr adaptor(enum_type v) : index(get_index(v)) {}
};
/*template <typename, typename> class adaptor;

template <typename enum_type_a, enum_type_a ... enum_values_a,
          typename enum_type_b, enum_type_b ... enum_values_b>
class adaptor<enum_container<enum_type_a, enum_values_a...>,
              enum_container<enum_type_b, enum_values_b...>> {
private:
	using container_a = enum_container<enum_type_a, enum_values_a...>;
	using container_b = enum_container<enum_type_b, enum_values_b...>;
	static_assert(container_a::size() == container_b::size(),
			"enums must have the same number of elements");
	static constexpr std::size_t sz = container_a::size();
	unsigned int index;

	static constexpr unsigned int
	get_index(enum_type_a value)
	{
		for (unsigned int i = 0; i < size(); ++i)
			if (container_a::values[i] == value)
				return i;
		throw bad_enum_value();
	}

	static constexpr unsigned int
	get_index(enum_type_b value)
	{
		for (unsigned int i = 0; i < size(); ++i)
			if (container_b::values[i] == value)
				return i;
		throw bad_enum_value();
	}
public:
	static constexpr std::size_t size() { return sz; }
	constexpr operator enum_type_a() const { return container_a::values[index]; }
	constexpr operator enum_type_b() const { return container_b::values[index]; }
	constexpr bool operator==(const adaptor& o) { return index == o.index; }
	constexpr bool operator==(enum_type_a v) { return index == get_index(v); }
	constexpr bool operator==(enum_type_b v) { return index == get_index(v); }
	constexpr bool operator!=(const adaptor& o) { return !operator==(o); }
	constexpr bool operator!=(enum_type_a v) { return !operator==(v); }
	constexpr bool operator!=(enum_type_b v) { return !operator==(v); }

	constexpr adaptor(enum_type_a v) : index(get_index(v)) {}
	constexpr adaptor(enum_type_b v) : index(get_index(v)) {}
};*/

}
