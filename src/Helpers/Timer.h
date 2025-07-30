#pragma once

#include <iostream>
#include <chrono>
#include <string>

class Timer {
public:
	Timer() : m_name(""), start(std::chrono::high_resolution_clock::now()) {}
	Timer(const std::string& name) : m_name(name), start(std::chrono::high_resolution_clock::now()) {}

	~Timer() {
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		//std::cout << "Timer [" << m_name << "]: " << duration << " microseconds.\n";
	}

private:
	std::string m_name;
	std::chrono::time_point<std::chrono::high_resolution_clock> start;
};