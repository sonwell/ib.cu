#pragma once
#include <ostream>
#include <functional>

namespace util {

template <typename wrapped_type>
class getset {
protected:
	using value_type = wrapped_type;
	using getter_type = std::function<value_type(void)>;
	using setter_type = std::function<void(const value_type&)>;

	getter_type getter;
	setter_type setter;
public:
	operator value_type() const { return getter(); }
	getset& operator=(const value_type& v) { setter(v); return *this; }

	getset(getter_type g, setter_type s) :
		getter(g), setter(s) {}
};

template <typename wrapped_type>
inline std::ostream&
operator<<(std::ostream& out, const getset<wrapped_type>& w)
{
	return out << (wrapped_type) w;
}

template <typename wrapped_type>
class cached {
protected:
	using value_type = wrapped_type;
	using setter_type = std::function<void(const value_type&)>;
private:
	setter_type setter;
	value_type value;
public:
	operator value_type() const { return value; }
	cached& operator=(const value_type& v)
	{
		value = v;
		setter(v);
		return *this;
	}

	cached(setter_type s, value_type v) :
		setter(s), value(v) {}
};

template <typename wrapped_type>
inline std::ostream&
operator<<(std::ostream& out, const cached<wrapped_type>& w)
{
	return out << (wrapped_type) w;
}

}
