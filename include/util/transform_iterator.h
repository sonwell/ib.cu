#include <iterator>

template <typename iterator_type, typename transform>
struct transform_iterator {
	transform f;
	iterator_type iter;

	using iter_value_type = typename std::iterator_traits<iterator_type>::value_type;
	using value_type = decltype(f(std::declval<iter_value_type>()));

	bool operator==(const transform_iterator& other) const
	{ return iter == other.iter; }
	bool operator!=(const transform_iterator& other) const
	{ return !operator==(other); }

	transform_iterator operator++(int)
	{ transform_iterator copy(*this); ++iter; return copy; }
	transform_iterator operator--(int)
	{ transform_iterator copy(*this); --iter; return copy; }
	transform_iterator& operator++() { ++iter; return *this; }
	transform_iterator& operator--() { --iter; return *this; }

	value_type operator*() const { return f(*iter); }
	value_type operator[](int n) const { return f(iter[n]); }

	transform_iterator(transform f, iterator_type iter) :
		f(f), iter(iter) {}
};
