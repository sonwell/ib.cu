#pragma once
#include <type_traits>
#include <ostream>
#include <vector>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include "util/getset.h"
#include "util/device.h"
#include "exceptions.h"
#include "memory_resource.h"
#include "scope.h"

namespace cuda {

enum class compute_mode : std::underlying_type_t<compute_mode_t> {
	default_mode = cudaComputeModeDefault,
	exclusive = cudaComputeModeExclusive,
	prohibited = cudaComputeModeProhibited,
	exclusive_process = cudaComputeModeExclusiveProcess
};

inline std::ostream&
operator<<(std::ostream& out, const compute_mode& cm)
{
	switch (cm) {
		case compute_mode::exclusive:
			return out << "exclusive";
		case compute_mode::prohibited:
			return out << "prohibited";
		case compute_mode::exclusive_process:
			return out << "exclusive process";
		default:
			return out << "default";
	}
}

enum class schedule : unsigned int {
	automatic = cudaDeviceScheduleAuto, spin = cudaDeviceScheduleSpin,
	yield = cudaDeviceScheduleYield,
	blocking_sync = cudaDeviceScheduleBlockingSync
};

inline std::ostream&
operator<<(std::ostream& out, const schedule& strategy)
{
	switch (strategy) {
		case schedule::spin:
			return out << "spin";
		case schedule::yield:
			return out << "yield";
		case schedule::blocking_sync:
			return out << "blocking sync";
		default:
			return out << "automatic";
	}
}

class device_flags {
private:
	schedule strategy;
	bool map_host;
	bool resize_local_mem_to_max;
public:
	static constexpr auto map_host_mask = cudaDeviceMapHost;
	static constexpr auto resize_mem_mask = cudaDeviceLmemResizeToMax;
	static constexpr auto schedule_mask = cudaDeviceScheduleMask;

	operator unsigned int() const
	{
		return static_cast<unsigned int>(strategy) +
			map_host_mask * map_host +
			resize_mem_mask * resize_local_mem_to_max;
	}

	device_flags(schedule strategy, bool map_host,
			bool resize_local_mem_to_max) :
		strategy(strategy), map_host(map_host),
		resize_local_mem_to_max(resize_local_mem_to_max) {}

	device_flags(unsigned int flags) :
		strategy(static_cast<schedule>(flags & schedule_mask)),
		map_host(flags & map_host_mask),
		resize_local_mem_to_max(flags & resize_mem_mask) {}

	friend std::ostream&
	operator<<(std::ostream& out, const device_flags& flags)
	{
		return out << "schedule strategy: " << flags.strategy << ", "
		           << "map host: " << (flags.map_host ? "yes" : "no") << ", "
		           << "result local memory to max"
		           << (flags.resize_local_mem_to_max ? "yes" : "no");
	}
};

struct compute_capability { int major, minor; };

inline std::ostream&
operator<<(std::ostream& out, const compute_capability& cc)
{
	return out << cc.major << '.' << cc.minor;
}

class device;
void synchronize(const device&);
void reset(const device&);
void set_device(const device&);
device& get_device();

class device : public util::device {
private:
	device_flags
	get_flags() const
	{
		unsigned int flags;
		scope lock(*this);
		throw_if_error(cudaGetDeviceFlags(&flags),
				"could not set device flags");
		return flags;
	}

	void
	set_flags(const device_flags& flags)
	{
		scope lock(*this);
		throw_if_error(cudaSetDeviceFlags(flags),
				"could not set device flags");
	}

	struct private_key {};

	static auto
	devices()
	{
		int device_count;
		std::vector<device> device_list;

		throw_if_error(cudaGetDeviceCount(&device_count),
				"could not get device count");
		device_list.reserve(device_count);
		for (int i = 0; i < device_count; ++i)
			device_list.emplace_back(i, private_key{});
		return device_list;
	}
protected:
	static device_prop_t
	props(int id)
	{
		device_prop_t props;
		throw_if_error(cudaGetDeviceProperties(&props, id),
				"could not get device properties");
		return props;
	}

	device(int id, const device_prop_t& props) :
		id(id),
		name(props.name),
		total_global_memory(props.totalGlobalMem),
		max_threads_per_block(props.maxThreadsPerBlock),
		max_block_dim{props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]},
		max_grid_dim{props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]},
		max_shared_memory_per_block(props.sharedMemPerBlock),
		total_constant_memory(props.totalConstMem),
		warp_size(props.warpSize),
		max_pitch(props.memPitch),
		max_registers_per_block(props.regsPerBlock),
		clock_rate(props.clockRate),
		gpu_overlap(props.deviceOverlap),
		multiprocessor_count(props.multiProcessorCount),
		kernel_exec_timeout(props.kernelExecTimeoutEnabled),
		integrated(props.integrated),
		can_map_host_memory(props.canMapHostMemory),
		compute_mode(static_cast<cuda::compute_mode>(props.computeMode)),
		concurrent_kernels(props.concurrentKernels),
		ecc_enabled(props.ECCEnabled),
		pci_bus_id(props.pciBusID),
		pci_device_id(props.pciDeviceID),
		tcc_driver(props.tccDriver),
		memory_clock_rate(props.memoryClockRate),
		global_memory_bus_width(props.memoryBusWidth),
		l2_cache_size(props.l2CacheSize),
		max_threads_per_multiprocessor(props.maxThreadsPerMultiProcessor),
		unified_addressing(props.unifiedAddressing),
		compute_capability{props.major, props.minor},
		stream_priorities_supported(props.streamPrioritiesSupported),
		global_l1_cache_supported(props.globalL1CacheSupported),
		local_l1_cache_supported(props.localL1CacheSupported),
		max_shared_memory_per_multiprocessor(props.sharedMemPerMultiprocessor),
		max_registers_per_multiprocessor(props.regsPerMultiprocessor),
		managed_memory(props.managedMemory),
		is_multi_gpu_board(props.isMultiGpuBoard),
		multi_gpu_board_group_id(props.multiGpuBoardGroupID),
		host_native_atomic_supported(props.hostNativeAtomicSupported),
		single_to_double_precision_perf_ratio(props.singleToDoublePrecisionPerfRatio),
		pageable_memory_access(props.pageableMemoryAccess),
		concurrent_managed_access(props.concurrentManagedAccess),
		compute_preemption_supported(props.computePreemptionSupported),
		can_use_host_pointer_for_registered_mem(props.canUseHostPointerForRegisteredMem),
		cooperative_launch(props.cooperativeLaunch),
#if CUDART_VERSION > 9000
		pageable_memory_access_uses_host_page_tables(props.pageableMemoryAccessUsesHostPageTables),
		direct_managed_mem_access_from_host(props.directManagedMemAccessFromHost),
#else
		pageable_memory_access_uses_host_page_tables(false),
		direct_managed_mem_access_from_host(false),
#endif
		flags([&] () { return get_flags(); }, [&] (const device_flags& flags) { set_flags(flags); }),
		resource(*this) {}
	device(int id) : device(id, props(id)) {}
public:
	static std::vector<device>& all() { static auto list = devices(); return list; }
	void synchronize() { ::cuda::synchronize(*this); }
	void reset() { ::cuda::reset(*this); }
	virtual util::memory_resource* memory() { return &resource; }

	device(int id, private_key) : device(id) {}
	virtual ~device() = default;

	const int id;
	const std::string name;
	const std::size_t total_global_memory;
	const int max_threads_per_block;
	const int max_block_dim[3];
	const int max_grid_dim[3];
	const int max_shared_memory_per_block;
	const std::size_t total_constant_memory;
	const int warp_size;
	const int max_pitch;
	const int max_registers_per_block;
	const int clock_rate;
	const bool gpu_overlap;
	const int multiprocessor_count;
	const int kernel_exec_timeout;
	const bool integrated;
	const bool can_map_host_memory;
	const cuda::compute_mode compute_mode;
	const bool concurrent_kernels;
	const bool ecc_enabled;
	const int pci_bus_id;
	const int pci_device_id;
	const int tcc_driver;
	const int memory_clock_rate;
	const int global_memory_bus_width;
	const int l2_cache_size;
	const int max_threads_per_multiprocessor;
	const bool unified_addressing;
	const cuda::compute_capability compute_capability;
	const bool stream_priorities_supported;
	const bool global_l1_cache_supported;
	const bool local_l1_cache_supported;
	const int max_shared_memory_per_multiprocessor;
	const int max_registers_per_multiprocessor;
	const int managed_memory;
	const bool is_multi_gpu_board;
	const int multi_gpu_board_group_id;
	const bool host_native_atomic_supported;
	const int single_to_double_precision_perf_ratio;
	const bool pageable_memory_access;
	const bool concurrent_managed_access;
	const bool compute_preemption_supported;
	const bool can_use_host_pointer_for_registered_mem;
	const bool cooperative_launch;
	const bool pageable_memory_access_uses_host_page_tables;
	const bool direct_managed_mem_access_from_host;
	util::getset<device_flags> flags;
private:
	device_memory resource;
};

inline std::ostream&
operator<<(std::ostream& out, const device& d)
{
	int clock_rate = d.clock_rate;
	int global_memory = d.total_global_memory;

	out << "device " << d.id << " (" << d.name << "):" << std::endl;
	out << "  compute capability: " << d.compute_capability << std::endl;
	if (global_memory > (1<<30))
		out << "  total global memory: " << (global_memory/1000000000.) << "GB" << std::endl;
	else if (global_memory > (1<<20))
		out << "  total global memory: " << (global_memory/1000000.) << "MB" << std::endl;
	else if (global_memory > (1<<10))
		out << "  total global memory: " << (global_memory/1000.) << "KB" << std::endl;
	else
		out << "  total global memory: " << global_memory << "B" << std::endl;
		out << "  total constant memory: " << d.total_constant_memory << std::endl;
		out << "  global memory bus width: " << d.global_memory_bus_width << " bits" << std::endl;
		out << "  number of multiprocessors: " << d.multiprocessor_count << std::endl;
		if (clock_rate > 1000000)
			out << "  clock rate: " << clock_rate / 1000000. << "GHz" << std::endl;
		else if (clock_rate > 1000)
			out << "  clock rate: " << clock_rate / 1000. << "MHz" << std::endl;
		else
			out << "  clock rate: " << clock_rate << "KHz" << std::endl;
		out << "  compute mode: " << d.compute_mode << std::endl;
		out << "  flags: " << d.flags << std::endl;
		out << "  supports unified addressing: " << d.unified_addressing << std::endl;
		out << "  warp size: " << d.warp_size;
		return out;
	}

	inline void
	synchronize()
	{
		cudaDeviceSynchronize();
	}

	inline void
	synchronize(const device& d)
	{
		scope lock(d);
		synchronize();
	}

	inline void
	reset()
	{
		cudaDeviceReset();
	}

	inline void
	reset(const device& d)
	{
		scope lock(d);
		reset();
	}

	inline auto&
	devices()
	{
		return device::all();
	}

	inline device&
	default_device()
	{
		return devices()[0];
	}

	inline void
	set_device(const device& d)
	{
		throw_if_error(cudaSetDevice(d.id),
				"could not set device id");
}

inline device&
get_device()
{
	int id;
	throw_if_error(cudaGetDevice(&id),
			"could not get device id");
	return devices()[id];
}

}
