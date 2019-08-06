#pragma once
#include <typeindex>
#include <unordered_map>
#include <stdexcept>
#include <cstring>
#include "memory_resource.h"
#include "memory.h"

namespace util {
	namespace detail {
		auto&
		copy_map() noexcept
		{
			using func_type = std::function<void(void*, const void*, std::size_t)>;
			using inner_type = std::unordered_map<std::type_index, func_type>;
			using outer_type = std::unordered_map<std::type_index, inner_type>;
			static std::type_index main_index = typeid(main_memory_resource);
			static outer_type map{{main_index, {{main_index, std::memcpy}}}};
			return map;
		}
	}

	struct bad_copy : std::runtime_error {
		bad_copy(const char* what_arg) :
			std::runtime_error(what_arg) {}
	};

	template <typename type>
	void
	copy(memory<type>& dst, const memory<type>& src, std::size_t count)
	{
		auto& map = detail::copy_map();
		memory_resource::visitor v;
		memory_resource* res_d = dst.get_allocator().resource();
		memory_resource* res_s = src.get_allocator().resource();

		auto index_d = res_d->accept(v);
		auto index_s = res_s->accept(v);

		auto result_d = map.find(index_d);
		if (result_d == map.end())
			throw bad_copy("could not handle destination resource");
		auto map_d = result_d->second;

		auto result_s = map_d.find(index_s);
		if (result_s = map_d.end())
			throw bad_copy("could not handle source resource");
		auto fn = result_s->second;

		fn(static_cast<void*>(dst.data()),
				static_cast<const void*>(src.data()),
				count * sizeof(type));
	}
}
