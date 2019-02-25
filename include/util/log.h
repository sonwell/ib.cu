#pragma once
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdio>
#include "functional.h"

namespace util {
	enum class log_level {
		notset = 0,
		debug = 10,
		info = 20,
		warn = 30,
		error = 40,
		critical = 50
	};

	struct logfile {
		virtual void write(const std::string&) = 0;
		virtual ~logfile() {}
	};

	struct stream_logfile : logfile {
		std::ostream& stream;

		virtual void
		write(const std::string& str)
		{
			stream << str;
		}

		stream_logfile(std::ostream& stream) :
			stream(stream) {}
	};

	struct file_logfile : logfile {
		std::FILE* file;

		virtual void
		write(const std::string& str)
		{
			using value_type = typename std::string::value_type;
			fwrite(str.c_str(), sizeof(value_type), str.size(), file);
		}

		file_logfile(const char* filename) :
			file(fopen(filename, "a")) {}
		~file_logfile() { fclose(file); }
	};

	struct logger {
		logfile* file;
		log_level min_level;

		static const char*
		level_as_str(log_level level)
		{
			switch (level) {
				case log_level::notset: return "not set";
				case log_level::debug: return "debug";
				case log_level::info:  return "info";
				case log_level::warn:  return "warn";
				case log_level::error: return "error";
				case log_level::critical: return "critical";
			};
			return "unknown";
		}

		template <typename ... arg_types>
		void
		write(log_level level, arg_types&& ... args)
		{
			using functional::map;
			if (level < min_level)
				return;

			std::stringstream ss;
			auto now = time(nullptr);
			auto* tm = localtime(&now);
			char* asc = asctime(tm);
			for (int i = 0; i < 24; ++i)
				ss << asc[i];
			ss << ": (" << level_as_str(level) << ") ";
			auto w = [&] (auto&& arg) { ss << arg; };
			map(w, std::forward_as_tuple(std::forward<arg_types>(args)...));
			ss << std::endl;
			file->write(ss.str());
		}

		logger(logfile& file, log_level min_level) :
			file(&file), min_level(min_level) {}
	};

	namespace impl {
		logger*&
		logger_storage()
		{
			static stream_logfile default_logfile(std::cerr);
			static logger default_logger(default_logfile, log_level::info);
			static logger* log_ptr = &default_logger;
			return log_ptr;
		}
	}

	logger*
	get_logger()
	{
		return impl::logger_storage();
	}

	void
	set_logger(logger& log)
	{
		impl::logger_storage() = &log;
	}

	template <typename ... arg_types>
	void
	log(log_level level, arg_types&& ... args)
	{
		logger* log = get_logger();
		log->write(level, std::forward<arg_types>(args)...);
	}

	namespace logging {
		template <typename ... arg_types>
		void
		debug(arg_types&& ... args)
		{
			log(log_level::debug, std::forward<arg_types>(args)...);
		}

		template <typename ... arg_types>
		void
		info(arg_types&& ... args)
		{
			log(log_level::info, std::forward<arg_types>(args)...);
		}

		template <typename ... arg_types>
		void
		warn(arg_types&& ... args)
		{
			log(log_level::warn, std::forward<arg_types>(args)...);
		}

		template <typename ... arg_types>
		void
		error(arg_types&& ... args)
		{
			log(log_level::error, std::forward<arg_types>(args)...);
		}

		template <typename ... arg_types>
		void
		critical(arg_types&& ... args)
		{
			log(log_level::critical, std::forward<arg_types>(args)...);
		}
	}
}
