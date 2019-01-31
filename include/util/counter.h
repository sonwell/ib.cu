#include <cstddef>

namespace util {
	// XXX this crap will probably not compile eventually:
	// http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#1850
	namespace counter_impl {
		template <std::size_t counter_id, std::size_t counter_state>
		struct flag {
			// These will silence gcc, clang, and icpc
			// It is correct that adl_flag is a non-template friend

			// Store old gcc diagnostics (clang will respect this)
			#pragma GCC diagnostic push
			// Silence icpc
			#pragma GCC diagnostic ignored "-Wpragmas"
			// gcc does not recognize "unknown warning option" group, silence it
			#pragma GCC diagnostic ignored "-Wunknown-warning-option"
			// Silence gcc
			#ifdef __INTEL_COMPILER  // Can't seem to silence gcc otherwise...
			#pragma warning (disable:1624)
			#endif
			// clang does not recognize "non template friend" group, silence it
			#pragma GCC diagnostic ignored "-Wnon-template-friend"
			// This does not produce a warning in clang
			// gcc errs on the side of caution since it's a bug 50% of the time
			// icpc copies gcc's behavior

			// *Declare* adl_flag for flag<counter_id, counter_state> (see: L38)
			friend constexpr std::size_t adl_flag(flag);
			// Restore icpc options
			#ifdef __INTEL_COMPILER
			#pragma warning (enable:1624)
			#endif
			// Restore old gcc diagnostics
			#pragma GCC diagnostic pop
		};

		template <std::size_t counter_id, std::size_t counter_state>
		struct writer {
			// *Define* adl_flag for flag<T, I, N> (see: L26)
			friend constexpr std::size_t
				adl_flag(flag<counter_id, counter_state>) { return counter_state; }

			// Need something to force instantiation of writer
			static constexpr std::size_t value = counter_state;
			static constexpr std::size_t id = counter_id;
		};

		/* ADL failed: this is the state of the counter */
		template <std::size_t counter_id, std::size_t counter_state>
		constexpr std::size_t
		reader(float, flag<counter_id, counter_state>)
		{
			return counter_state;
		}

		/* ADL succeeded: continue the search in result */
		template <std::size_t counter_id, std::size_t counter_state,
				 std::size_t = adl_flag(flag<counter_id, counter_state>{}),  // SFINAE
				 std::size_t result = reader(0, flag<counter_id, counter_state+1>{})>
		constexpr std::size_t
		reader(int, flag<counter_id, counter_state>, std::size_t state = result)
		{
			return state;
		}

		template <typename, std::size_t> class counter;

		/* Increment the counter and return the value */
		template <typename T, std::size_t counter_id,
				 std::size_t counter_state = reader(0, flag<counter_id, 0>{})>
		constexpr T
		next(const counter<T, counter_id>& c,
				std::size_t state = writer<counter_id, counter_state>::value)
		{
			return c.start + c.by * state;
		}

		template <>
		struct counter<std::size_t, 0> {
			using type = std::size_t;
			static constexpr std::size_t id = 0;
			const std::size_t start;
			const std::size_t by;

			constexpr counter() : start(0), by(1) {}
		friend constexpr std::size_t
			adl_flag(flag<id, 0>) { return 0; }
		};

		template <typename>
		struct zero {
			static constexpr auto value = 0ull;
		};

		template <typename counter_type, std::size_t counter_id =
			next(counter<std::size_t, zero<counter_type>::value>())>
		struct counter {
			using type = counter_type;
			static constexpr std::size_t id = counter_id;
			const type start;
			const type by;

			constexpr counter(type start = 0, type by = 1) :
				start(start), by(by) {}
		};

		// XXX can't get the deduction guides to actually advance the counter
		/*
		template <typename type> counter(type, type) -> counter<type, next(counter<std::size_t, zero<type>::value>())>;
		template <typename type> counter(type) -> counter<type, next(counter<std::size_t, zero<type>::value>())>;
		template <typename type=void> counter() -> counter<std::size_t, next(counter<std::size_t, zero<type>::value>())>;
		*/
	}

	using counter_impl::counter;
}

namespace std {
	using util::counter_impl::next;
}
