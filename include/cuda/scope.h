#pragma once

namespace cuda {
	class device;
	void set_device(const device&);
	device& get_device();

	class scope {
	private:
		const device& d;
	public:
		scope(const device& dev) :
			d(get_device()) { set_device(dev); }
		~scope() { set_device(d); }
	};
}
