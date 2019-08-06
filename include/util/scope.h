#pragma once

namespace util {
	struct scope {
	private:
		std::function<void(void)> f;
		scope& swap(scope& o) { std::swap(f, o.f); return *this; }
	public:
		scope() : f([] () {}) {}
		scope(std::function<void(void)> f) : f(f) {}
		scope(const scope&) = delete;
		scope(scope&& o) : f([] () {}) { swap(o); }
		~scope() { f(); }
	};
}
