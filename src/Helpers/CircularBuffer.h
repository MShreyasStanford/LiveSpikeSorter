#ifndef CIRCULAR_BUFFER_H_
#define CIRCULAR_BUFFER_H_

#include <algorithm>
#include <vector>
#include <stdexcept>

template<typename T>
class CircularBuffer {
public:
	CircularBuffer(size_t capacity)
		: m_capacity(capacity), m_size(0), m_start(0), m_data(capacity) {}

	void push_back(const T& value) {
		if (m_size < m_capacity) {
			size_t index = (m_start + m_size) % m_capacity;
			m_data[index] = value;
			m_size++;
		}
		else {
			m_data[m_start] = value;
			m_start = (m_start + 1) % m_capacity;
		}
	}

	T& operator[](size_t index) {
		if (index >= m_size) throw std::out_of_range("Index out of range");
		return m_data[(m_start + index) % m_capacity];
	}

	const T& operator[](size_t index) const {
		if (index >= m_size) throw std::out_of_range("Index out of range");
		return m_data[(m_start + index) % m_capacity];
	}

	size_t size() const {
		return m_size;
	}

	bool empty() const {
		return m_size == 0;
	}

	void clear() {
		m_size = 0;
		m_start = 0;
	}

	void resize(size_t new_size, const T& value = T()) {
		if (new_size > m_capacity) throw std::length_error("New size exceeds capacity");
		for (size_t i = m_size; i < new_size; ++i) {
			(*this).push_back(value);
		}
		m_size = new_size;
	}

	typename std::vector<T>::iterator begin() {
		return m_data.begin() + m_start;
	}

	typename std::vector<T>::iterator end() {
		if (m_start + m_size <= m_capacity)
			return m_data.begin() + m_start + m_size;
		else
			return m_data.begin() + (m_start + m_size) % m_capacity;
	}

	typename std::vector<T>::const_iterator begin() const {
		return m_data.begin() + m_start;
	}

	typename std::vector<T>::const_iterator end() const {
		if (m_start + m_size <= m_capacity)
			return m_data.begin() + m_start + m_size;
		else
			return m_data.begin() + (m_start + m_size) % m_capacity;
	}

	operator std::vector<T>() const {
		std::vector<T> result;
		result.reserve(m_size);
		for (size_t i = 0; i < m_size; ++i) {
			result.push_back((*this)[i]);
		}
		return result;
	}

	typename std::vector<T>::reverse_iterator rbegin() {
		return std::vector<T>(*this).rbegin();
	}

	typename std::vector<T>::reverse_iterator rend() {
		return std::vector<T>(*this).rend();
	}

	typename std::vector<T>::const_reverse_iterator rbegin() const {
		return std::vector<T>(*this).rbegin();
	}

	typename std::vector<T>::const_reverse_iterator rend() const {
		return std::vector<T>(*this).rend();
	}


private:
	size_t m_capacity;
	size_t m_size;
	size_t m_start;
	std::vector<T> m_data;
};

#endif // CIRCULAR_BUFFER_H_