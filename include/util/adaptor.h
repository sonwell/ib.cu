#pragma once
#include <ostream>
#include <stdexcept>

namespace util {

struct bad_enum_value : public std::runtime_error {
	bad_enum_value() : std::runtime_error("bad enum value") {}
	bad_enum_value(const char* what_arg) :
		std::runtime_error(what_arg) {}
	bad_enum_value(std::string& what_arg) :
		std::runtime_error(what_arg) {}
};

template <typename enum_type, enum_type ... enum_values>
struct enum_container {
	using value_type = enum_type;
	static constexpr value_type values[] = { enum_values... };
	static constexpr std::size_t size() { return sizeof...(enum_values); }
	enum_container() = delete;
};

template <typename from_container, typename to_container>
class adaptor {
private:
	static_assert(from_container::size() == to_container::size(),
			"enums must have the same number of elements");
	using enum_type = typename from_container::value_type;
	using alias_type = typename to_container::value_type;
	static constexpr std::size_t sz = to_container::size();
	alias_type value;

	static constexpr enum_type
	map(alias_type value)
	{
		for (int i = 0; i < size(); ++i)
			if (to_container::values[i] == value)
				return from_container::values[i];
		throw bad_enum_value();
	}

	static constexpr alias_type
	map(enum_type value)
	{
		for (int i = 0; i < size(); ++i)
			if (from_container::values[i] == value)
				return to_container::values[i];
		throw bad_enum_value();
	}
public:
	static constexpr std::size_t size() { return sz; }
	constexpr operator enum_type() const { return map(value); }
	constexpr operator alias_type() const { return value; }
	constexpr bool operator==(const adaptor& o) { return value == o.value; }
	constexpr bool operator==(alias_type v) { return value == v; }
	constexpr bool operator==(enum_type v) { return map(value) == v; }
	constexpr bool operator!=(const adaptor& o) { return !operator==(o); }
	constexpr bool operator!=(alias_type v) { return !operator==(v); }
	constexpr bool operator!=(enum_type v) { return !operator==(v); }

	constexpr adaptor(alias_type value) : value(value) {}
	constexpr adaptor(enum_type value) : value(map(value)) {}
};

template <typename from_container, typename to_container>
inline std::ostream&
operator<<(std::ostream& out, const adaptor<from_container, to_container>& adr)
{
	return out << (typename to_container::value_type) adr;
}

}
