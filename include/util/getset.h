#pragma once
#include <functional>

namespace util {
	template <typename wrapped_type>
	class getset {
	protected:
		using value_type = wrapped_type;
		using getter_type = std::function<value_type(void)>;
		using setter_type = std::function<void(const wrapped_type&)>;

		getter_type getter;
		setter_type setter;
	public:
		operator value_type() const { return getter(); }
		getset& operator=(const value_type& v) { setter(v); return *this; }

		getset(getter_type g, setter_type s) :
			getter(g), setter(s) {}
		getset(getset&& o) :
			getter(o.getter),
			setter(o.setter) {}
	};

	template <typename container_type>
	class cached {
	protected:
		using value_type = typename container_type::value_type;
		using getter_type = typename container_type::getter_type;
		using setter_type = typename container_type::setter_type;
	private:
		container_type container;
		value_type value;
	public:
		operator value_type() const { return value; }
		cached& operator=(const value_type& v)
		{
			container = v;
			value = v;
			return *this;
		}

		cached(container_type&& container) :
			container(std::move(container)),
			value((value_type) container) {}
		template <typename ... arg_types>
		cached(arg_types&& ... args) :
			container(std::forward<arg_types>(args)...),
			value((value_type) container) {}
		cached(cached&& o) :
			container(std::move(o.container)),
			value(std::move(o.value)) {}
	};
}
