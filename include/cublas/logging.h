#pragma once

namespace cublas {
namespace logging {

struct configuration {
	bool log_to_stdout;
	bool log_to_stderr;
	const char* log_filename;

	constexpr configuration(
			bool log_to_stdout = false,
			bool log_to_stderr = false,
			const char* log_filename = nullptr) :
		log_to_stdout(log_to_stdout),
		log_to_stderr(log_to_stderr),
		log_filename(log_filename) {}
	configuration(std::FILE* fd) :
		configuration(fd == stdout, fd == stderr) {}
};

inline void
configure(const configuration& cfg)
{
	bool enable = cfg.log_to_stdout ||
		cfg.log_to_stderr ||
		cfg.log_filename != nullptr;
	cublasLoggerConfigure(enable, cfg.log_to_stdout,
			cfg.log_to_stderr, cfg.log_filename);
}

inline cublasLogCallback
get_callback()
{
	cublasLogCallback fn;
	cublasGetLoggerCallback(&fn);
	return fn;
}

inline void
set_callback(cublasLogCallback fn)
{
	cublasSetLoggerCallback(fn);
}

} // namespace logging
} // namespace cublas
