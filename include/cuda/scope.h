#pragma once

namespace cuda {
	class scope {
	private:
		int last_id;

		static int get()
		{
			int id;
			throw_if_error(cudaGetDevice(&id));
			return id;
		}

		static void set(int id)
		{
			throw_if_error(cudaSetDevice(id));
		}
	public:
		scope(int id) : last_id(get()) { set(id); }
		~scope() { set(last_id); }
	};
}
