#pragma once

#include <iostream>
#include <chrono>
#include <string>
#include <functional>
#include <utility>

class Timer {
public:
	using Callback = std::function<void(const std::string&, long long)>;

	Timer() : m_name(""), start(std::chrono::high_resolution_clock::now()) {}
	Timer(const std::string& name) : m_name(name), start(std::chrono::high_resolution_clock::now()) {}

	static void SetThreadCallback(Callback callback) {
		threadCallbackRef() = std::move(callback);
	}

	static void ClearThreadCallback() {
		threadCallbackRef() = nullptr;
	}

	~Timer() {
		auto end = std::chrono::high_resolution_clock::now();
		long long duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		//std::cout << "Timer [" << m_name << "]: " << duration << " microseconds.\n";
		if (!m_name.empty() && threadCallbackRef()) {
			threadCallbackRef()(m_name, duration);
		}
	}

private:
	static Callback& threadCallbackRef() {
		static thread_local Callback callback;
		return callback;
	}

	std::string m_name;
	std::chrono::time_point<std::chrono::high_resolution_clock> start;
};
