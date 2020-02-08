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
	static_assert(std::is_enum<enum_type>::value,
			"enum_container only allows enum types");
	using value_type = enum_type;
	static constexpr value_type values[] = { enum_values... };
	static constexpr std::size_t size() { return sizeof...(enum_values); }
	enum_container() = delete;
};

template <typename, typename> class adaptor;

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
};

}
