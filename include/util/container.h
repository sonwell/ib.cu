#pragma once

namespace util {

template <typename ...> struct container {};
template <typename ... types>
container(types&&...) -> container<types...>;

template <typename ... types>
constexpr auto
concat(container<types...> c)
{
	return c;
}

template <typename ... left_types, typename ... right_types>
constexpr auto
concat(container<left_types...>, container<right_types...>)
{
	return container<left_types..., right_types...>{};
}

template <typename ... types, typename ... arg_types>
constexpr auto
concat(container<types...> c, arg_types ... args)
{
	return concat(c, concat(args...));
}

}
