#pragma once
#include <iostream>
#include "functional.h"

// listed as:
//   transform(namespace, class_name)
#define backends(transform, sep) \
	transform(cuda, context) sep \
	transform(mkl, context)

#define forward_declare(ns, name) \
	namespace ns { struct name; }
backends(forward_declare,);
#undef forward_declare

namespace util {
namespace detail {

template <typename T>
struct is_complete_helper {
    template <typename U>
    static auto test(U*)  -> std::integral_constant<bool, sizeof(U) == sizeof(U)>;
    static auto test(...) -> std::false_type;
    using type = decltype(test((T*)0));
};

template <typename T>
struct is_complete : is_complete_helper<T>::type {};

template <typename T>
inline constexpr auto is_complete_v = is_complete<T>::value;

template <int n>
struct error_thrower {
	template <int> struct confounder : std::false_type {};

	static_assert(confounder<n>::value,
			"Contexts use the visitor pattern to dispatch correctly. "
			"The compiler must therefore be aware of some type info to"
			"choose the correct function. Supply a namespace and class "
			"name in the *backends* macro in util/context.h .");
};

template <typename visitable>
struct visits { virtual void visit(const visitable&) const = 0; };

template <typename derived>
struct caller { template <typename visitable> struct calls; };

// C++20
//template <template <typename> typename tmpl, typename ... types>
//struct inherit : tmpl<types>... {
//    using tmpl<types>::visit...;
//    [[noreturn]] void visit(context&) const { throw an_error; }
//};

// Until the code base is C++20-compliant, do this:
template <template <typename> typename, typename ...> struct inherit;
template <template <typename> typename tmpl> struct inherit<tmpl> {};

template <template <typename> typename tmpl, typename first>
struct inherit<tmpl, first> : tmpl<first> { using tmpl<first>::visit; };

template <template <typename> typename tmpl, typename first, typename ... rest>
struct inherit<tmpl, first, rest...> : tmpl<first>, inherit<tmpl, rest...> {
	using tmpl<first>::visit;
	using inherit<tmpl, rest...>::visit;
};

#define list(ns, name) ns::name
#define comma ,
template <template <typename> typename tmpl>
using contexts = inherit<tmpl, backends(list, comma)>;
#undef list
#undef comma

} // namespace detail

class context {
public:
	template <typename> struct accepts;

	struct visitor : detail::contexts<detail::visits> {
		using detail::contexts<detail::visits>::visit;

		template <int n = 0>
		void visit(const context&) const
		{
			// reaching this means the supplied context
			// is not listed as a backend
			detail::error_thrower<n> thrower;
		}
	};
protected:
	virtual void accept(const visitor&) const = 0;

template <typename func_type, typename ... arg_types>
friend void visit(const context&, func_type&&, arg_types&& ...);
};

template <typename, typename ...> struct endpoint;

namespace detail {

template <typename derived>
template <typename visitable>
struct caller<derived>::calls : virtual context::visitor {
	using context::visitor::visit;

	virtual void
	visit(const visitable& v) const
	{
		// this should always be true but shuts up the
		// compiler about incomplete types.
		if constexpr (detail::is_complete_v<visitable>)
			static_cast<const derived*>(this)->call(v);
	}
};

template <typename func_type, typename ... arg_types>
using endpoint = contexts<caller<util::endpoint<func_type,
	  arg_types...>>::template calls>;

} // namespace detail

template <typename func_type, typename ... arg_types>
struct endpoint : detail::endpoint<func_type, arg_types...> {
	func_type fn;
	std::tuple<arg_types...> args;

	template <typename visited>
	void call(visited&& v) const
	{
		using namespace util::functional;
		apply(partial(fn, std::forward<visited>(v)), args);
	}

	constexpr endpoint(func_type fn, arg_types ... args) :
		fn(std::forward<func_type>(fn)),
		args{std::forward<arg_types>(args)...} {}
};

template <typename func_type, typename ... arg_types>
void visit(const context& ctx, func_type&& fn, arg_types&& ... args)
{
	ctx.accept(endpoint{std::forward<func_type>(fn),
			std::forward<arg_types>(args)...});
}

#define overload_visit(ns, name) \
template <typename func_type, typename ... arg_types> \
void visit(const ns::name& ctx, func_type&& fn, arg_types&& ... args) \
{ \
	fn(ctx, std::forward<arg_types>(args)...); \
}
backends(overload_visit,);
#undef overload_visit

template <typename derived>
struct context::accepts : public context {
	virtual void
	accept(const visitor& v) const
	{
		v.visit(*static_cast<const derived*>(this));
	}
};

}

#undef backends
