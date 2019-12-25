#pragma once
#include <iostream>
#include <ostream>
#include <functional>
#include <type_traits>

namespace util {
namespace detail {

template <typename T>
struct preincrementable {
	template <typename U> static auto test(U& u, int) -> std::is_void<std::void_t<decltype(++u)>>;
	template <typename U> static auto test(U& u, float) -> std::false_type;
	static constexpr auto value = decltype(test(std::declval<T&>(), 0))::value;
};

template <typename T>
struct postincrementable {
	template <typename U> static auto test(U& u, int) -> std::is_void<std::void_t<decltype(u++)>>;
	template <typename U> static auto test(U& u, float) -> std::false_type;
	static constexpr auto value = decltype(test(std::declval<T&>(), 0))::value;
};

template <typename T>
struct predecrementable {
	template <typename U> static auto test(U& u, int) -> std::is_void<std::void_t<decltype(--u)>>;
	template <typename U> static auto test(U& u, float) -> std::false_type;
	static constexpr auto value = decltype(test(std::declval<T&>(), 0))::value;
};

template <typename T>
struct postdecrementable {
	template <typename U> static auto test(U& u, int) -> std::is_void<std::void_t<decltype(u--)>>;
	template <typename U> static auto test(U& u, float) -> std::false_type;
	static constexpr auto value = decltype(test(std::declval<T&>(), 0))::value;
};

}

template <typename wrapped_type>
class getset {
protected:
	using value_type = std::remove_reference_t<wrapped_type>;
	using getter_type = std::function<wrapped_type(void)>;
	using setter_type = std::function<void(value_type)>;

	getter_type getter;
	setter_type setter;
public:
	template <typename cast_type>
		constexpr operator cast_type() const { return (cast_type) getter(); }
	constexpr operator wrapped_type() const { return getter(); }
	constexpr getset& operator=(value_type&& v) { setter(std::move(v)); return *this; }
	constexpr getset& operator=(const value_type& v) { setter(v); return *this; }

	constexpr getset(getter_type g, setter_type s) :
		getter(g), setter(s) {}
};

#define assignment(transform) \
	transform(+=) \
	transform(-=) \
	transform(*=) \
	transform(/=) \
	transform(%=) \
	transform(^=) \
	transform(&=) \
	transform(|=) \
	transform(>>=) \
	transform(<<=)
#define increment(transform) \
	transform(++, detail::preincrementable, detail::postincrementable) \
	transform(--, detail::predecrementable, detail::postdecrementable)
#define binary(op) \
template <typename wrapped_type, typename arg_type, \
		  typename = decltype(std::declval<wrapped_type&>() op std::declval<arg_type>())> \
decltype(auto) \
operator op(getset<wrapped_type>& wr, arg_type&& arg) \
{ \
	std::remove_reference_t<wrapped_type> v = wr; \
	v op std::forward<arg_type>(arg); \
	return wr = v; \
}
#define unary(op, pre, post) \
template <typename wrapped_type> \
constexpr std::enable_if_t<post<wrapped_type>::value, wrapped_type> \
operator op(getset<wrapped_type>& wr, int) \
{ \
	std::remove_reference_t<wrapped_type> v = wr; \
	auto w = v; \
	v op; \
	wr = v; \
	return w; \
} \
template <typename wrapped_type> \
constexpr std::enable_if_t<pre<wrapped_type>::value, wrapped_type> \
operator op(getset<wrapped_type>& wr) \
{ \
	std::remove_reference_t<wrapped_type> v = wr; \
	op v; \
	wr = v; \
	return v; \
}

assignment(binary)
increment(unary)

#undef unary
#undef binary
#undef increment
#undef assignment

#define operators(transform) \
	transform(+) \
	transform(-) \
	transform(*) \
	transform(/) \
	transform(%) \
	transform(^) \
	transform(&) \
	transform(|) \
	transform(>>) \
	transform(<<)
#define binary(op) \
template <typename wrapped_type, typename arg_type, \
	typename = decltype(std::declval<wrapped_type &>() op std::declval<arg_type>())> \
constexpr decltype(auto) \
operator op(getset<wrapped_type>& gs, arg_type&& arg) \
{ \
	wrapped_type wr = gs; \
	auto&& r = wr op std::forward<arg_type>(arg); \
	gs = wr; \
	return r; \
} \
template <typename wrapped_type, typename arg_type, \
	typename = decltype(std::declval<wrapped_type &>() op std::declval<arg_type>())> \
constexpr decltype(auto) \
operator op(const getset<wrapped_type>& gs, arg_type&& arg) \
{ \
	const wrapped_type wr = gs; \
	return wr op std::forward<arg_type>(arg); \
} \
template <typename wrapped_type, typename arg_type, \
	typename = decltype(std::declval<arg_type>() op std::declval<wrapped_type &>())> \
constexpr decltype(auto) \
operator op(arg_type&& arg, getset<wrapped_type>& gs) \
{ \
	wrapped_type wr = gs; \
	auto&& r = std::forward<arg_type>(arg) op wr; \
	gs = wr; \
	return r; \
} \
template <typename wrapped_type, typename arg_type, \
	typename = decltype(std::declval<arg_type>() op std::declval<wrapped_type &>())> \
constexpr decltype(auto) \
operator op(arg_type&& arg, const getset<wrapped_type>& gs) \
{ \
	const wrapped_type wr = gs; \
	return std::forward<arg_type>(arg) op wr; \
}

operators(binary)

#undef binary
#undef operators

template <typename wrapped_type,
	typename = decltype(~std::declval<wrapped_type>())>
constexpr decltype(auto)
operator ~(const getset<wrapped_type>& gs)
{
	return ~(wrapped_type) gs;
}

template <typename wrapped_type,
	typename = decltype(!std::declval<wrapped_type>())>
constexpr decltype(auto)
operator !(const getset<wrapped_type>& gs)
{
	return !(wrapped_type) gs;
}

template <typename wrapped_type>
class cached : public getset<wrapped_type> {
protected:
	using base = getset<wrapped_type>;
	using value_type = typename base::value_type;
	using setter_type = typename base::setter_type;
private:
	value_type value;
public:
	constexpr cached(setter_type s, value_type v) :
		base{
			[&] () { return value; },
			[&, s=s] (value_type v) { value = std::move(v); s(value); }
		},
		value(v) {}
	using base::operator=;

	constexpr cached(const cached&) = delete;
	constexpr cached(cached&&) = delete;
};

}
