#pragma once

#include <vector>
#include <string>
#include <map>
#include <initializer_list>
#include <stdexcept>
#include <numeric>
#include <functional>

struct CopyDataTag {};
struct ViewDataTag {};

constexpr CopyDataTag CopyData{};
constexpr ViewDataTag ViewData{};

template <typename NumericType>
class TensorWrapper
{
public:
	// Constructor that copies the data (owns it)
	TensorWrapper(CopyDataTag, const NumericType* data,
		const std::vector<std::string>& dimNames,
		const std::vector<long>& dimLens);

	// Constructor that creates a view over the data (does not copy)
	TensorWrapper(ViewDataTag, const NumericType* data,
		const std::vector<std::string>& dimNames,
		const std::vector<long>& dimLens);

	~TensorWrapper();

	// Overload the at() method for initializer_list
	NumericType& at(const std::initializer_list<std::pair<std::string, int>>& indices);
	const NumericType& at(const std::initializer_list<std::pair<std::string, int>>& indices) const;

	// Overload the at() method for vector
	NumericType& at(const std::vector<std::pair<std::string, int>>& indices);
	const NumericType& at(const std::vector<std::pair<std::string, int>>& indices) const;

	// Assert that provided dimensions match internal dimensions
	void assert_dimensions(const std::vector<std::string>& dims) const;

	// Performs a transposition of the named dimensions
	void swapDimensions(const std::string& dimName1, const std::string& dimName2);

	NumericType* data() {
		if (isOwning) return ownedData.data();
		else throw std::runtime_error("Cannot return mutable data from a non-owning TensorWrapper");
	}

	// Const version: always returns const data pointer
	const NumericType* data() const {
		return dataPtr;
	}

private:
	// For owning the data
	std::vector<NumericType> ownedData;

	// For non-owning view
	const NumericType* dataPtr;

	// Whether we own the data
	bool isOwning;

	// Common members
	std::vector<std::string> dimNames;
	std::vector<long>        dimLens;
	std::map<std::string, size_t> dimNameToIndex;

	// Private helper function to initialize dimension mappings
	void initializeDimensions();

	// Common implementation for at() method
	template <typename Container>
	NumericType& access_element(const Container& indices);

	template <typename Container>
	const NumericType& access_element(const Container& indices) const;
};

// Implementation

template <typename NumericType>
void TensorWrapper<NumericType>::initializeDimensions()
{
	// Build the map from dimension names to indices
	for (size_t i = 0; i < dimNames.size(); ++i)
	{
		dimNameToIndex[dimNames[i]] = i;
	}
}

template <typename NumericType>
TensorWrapper<NumericType>::TensorWrapper(CopyDataTag, const NumericType* inputData,
	const std::vector<std::string>& dimNames,
	const std::vector<long>& dimLens)
	: dimNames(dimNames), dimLens(dimLens), dataPtr(nullptr), isOwning(true)
{
	initializeDimensions();

	// Calculate the total number of elements
	size_t totalElements = 1;
	for (const auto& len : dimLens)
	{
		totalElements *= len;
	}


	// Copy data into ownedData
	ownedData.assign(inputData, inputData + totalElements);

	// Set dataPtr to point to ownedData
	dataPtr = ownedData.data();
}

template <typename NumericType>
TensorWrapper<NumericType>::TensorWrapper(ViewDataTag, const NumericType* inputData,
	const std::vector<std::string>& dimNames,
	const std::vector<long>& dimLens)
	: dataPtr(inputData), dimNames(dimNames), dimLens(dimLens), isOwning(false)
{
	initializeDimensions();
	// No data is copied; dataPtr points to inputData
}

template <typename NumericType>
TensorWrapper<NumericType>::~TensorWrapper()
{
	// No special cleanup needed; ownedData and other members will be destroyed automatically
}

// Overloaded at() method for std::initializer_list (Non-const version)
template <typename NumericType>
NumericType& TensorWrapper<NumericType>::at(const std::initializer_list<std::pair<std::string, int>>& indices)
{
	return access_element(indices);
}

// Overloaded at() method for std::initializer_list (Const version)
template <typename NumericType>
const NumericType& TensorWrapper<NumericType>::at(const std::initializer_list<std::pair<std::string, int>>& indices) const
{
	return access_element(indices);
}

// Overloaded at() method for std::vector (Non-const version)
template <typename NumericType>
NumericType& TensorWrapper<NumericType>::at(const std::vector<std::pair<std::string, int>>& indices)
{
	return access_element(indices);
}

// Overloaded at() method for std::vector (Const version)
template <typename NumericType>
const NumericType& TensorWrapper<NumericType>::at(const std::vector<std::pair<std::string, int>>& indices) const
{
	return access_element(indices);
}

// Private helper function (Non-const version)
template <typename NumericType>
template <typename Container>
NumericType& TensorWrapper<NumericType>::access_element(const Container& indices)
{
	// Initialize indices vector
	std::vector<int> idx(dimNames.size(), 0);
	std::vector<bool> specified(dimNames.size(), false);

	// Set indices based on provided pairs
	for (const auto& pair : indices)
	{
		const std::string& dimName = pair.first;
		int index = pair.second;

		auto it = dimNameToIndex.find(dimName);
		if (it == dimNameToIndex.end())
		{
			std::cout << "Invalid dimension name: " + dimName << std::endl;
			throw std::invalid_argument("Invalid dimension name: " + dimName);
		}

		size_t pos = it->second;

		if (specified[pos])
		{
			std::cout << "Duplicate index for dimension: " + dimName << std::endl;
			throw std::invalid_argument("Duplicate index for dimension: " + dimName);
		}

		specified[pos] = true;

		if (index < 0 || index >= dimLens[pos])
		{
			std::cout << "Index out of bounds for dimension: " + dimName << std::endl;
			throw std::out_of_range("Index out of bounds for dimension: " + dimName);
		}

		idx[pos] = index;
	}

	// Calculate the flat index
	size_t flatIndex = 0;
	size_t stride = 1;

	for (size_t i = 0; i < dimNames.size(); ++i)
	{
		flatIndex += idx[i] * stride;
		stride *= dimLens[i];
	}

	return const_cast<NumericType&>(dataPtr[flatIndex]);
}

// Private helper function (Const version)
template <typename NumericType>
template <typename Container>
const NumericType& TensorWrapper<NumericType>::access_element(const Container& indices) const
{
	// Initialize indices vector
	std::vector<int> idx(dimNames.size(), 0);
	std::vector<bool> specified(dimNames.size(), false);

	// Set indices based on provided pairs
	for (const auto& pair : indices)
	{
		const std::string& dimName = pair.first;
		int index = pair.second;

		auto it = dimNameToIndex.find(dimName);
		if (it == dimNameToIndex.end())
		{
			throw std::invalid_argument("Invalid dimension name: " + dimName);
		}

		size_t pos = it->second;

		if (specified[pos])
		{
			throw std::invalid_argument("Duplicate index for dimension: " + dimName);
		}

		specified[pos] = true;

		if (index < 0 || index >= dimLens[pos])
		{
			throw std::out_of_range("Index out of bounds for dimension: " + dimName);
		}

		idx[pos] = index;
	}

	// Calculate the flat index
	size_t flatIndex = 0;
	size_t stride = 1;

	for (size_t i = 0; i < dimNames.size(); ++i)
	{
		flatIndex += idx[i] * stride;
		stride *= dimLens[i];
	}

	return dataPtr[flatIndex];
}

template <typename NumericType>
void TensorWrapper<NumericType>::assert_dimensions(const std::vector<std::string>& dims) const
{
	if (dims.size() != dimNames.size())
	{
		throw std::invalid_argument("Dimension size mismatch. Expected " + std::to_string(dimNames.size()) +
			" dimensions, but got " + std::to_string(dims.size()) + ".");
	}

	for (size_t i = 0; i < dims.size(); ++i)
	{
		if (dims[i] != dimNames[i])
		{
			throw std::invalid_argument("Dimension name mismatch at position " + std::to_string(i) +
				": expected '" + dimNames[i] + "', but got '" + dims[i] + "'.");
		}
	}
}

template <typename NumericType>
void TensorWrapper<NumericType>::swapDimensions(const std::string& dimName1, const std::string& dimName2)
{
	std::cout << "Performing transpose" << std::endl;

	// Verify that both dimensions exist
	auto it1 = dimNameToIndex.find(dimName1);
	auto it2 = dimNameToIndex.find(dimName2);

	if (it1 == dimNameToIndex.end())
	{
		throw std::invalid_argument("Dimension not found: " + dimName1);
	}
	if (it2 == dimNameToIndex.end())
	{
		throw std::invalid_argument("Dimension not found: " + dimName2);
	}

	size_t pos1 = it1->second;
	size_t pos2 = it2->second;

	if (pos1 == pos2)
	{
		// Dimensions are the same, nothing to do
		return;
	}

	// Store old dimension lengths before swapping
	std::vector<long> oldDimLens = dimLens;

	// Swap dimNames and dimLens
	std::swap(dimNames[pos1], dimNames[pos2]);
	std::swap(dimLens[pos1], dimLens[pos2]);

	// Rebuild dimNameToIndex
	initializeDimensions();

	if (isOwning)
	{
		// We own the data, rearrange it

		// Compute old strides (column-major order)
		size_t N = dimLens.size();
		std::vector<size_t> oldStrides(N);
		oldStrides[0] = 1;
		for (size_t i = 1; i < N; ++i)
		{
			oldStrides[i] = oldStrides[i - 1] * oldDimLens[i - 1];
		}

		// Compute new strides (column-major order)
		std::vector<size_t> newStrides(N);
		newStrides[0] = 1;
		for (size_t i = 1; i < N; ++i)
		{
			newStrides[i] = newStrides[i - 1] * dimLens[i - 1];
		}

		// Calculate total elements
		size_t totalElements = 1;
		for (const auto& len : dimLens)
		{
			totalElements *= len;
		}

		// Allocate new data
		std::vector<NumericType> newData(totalElements);

		// Iterate over all elements to rearrange data
		for (size_t old_flat_index = 0; old_flat_index < totalElements; ++old_flat_index)
		{
			// Compute old multi-dimensional indices
			std::vector<size_t> idx(N);
			size_t tempIndex = old_flat_index;
			for (size_t i = 0; i < N; ++i)
			{
				idx[i] = (tempIndex / oldStrides[i]) % oldDimLens[i];
			}

			// Swap the indices of the specified dimensions
			std::swap(idx[pos1], idx[pos2]);

			// Compute new flat index
			size_t new_flat_index = 0;
			for (size_t i = 0; i < N; ++i)
			{
				new_flat_index += idx[i] * newStrides[i];
			}

			// Rearrange the data
			newData[new_flat_index] = ownedData[old_flat_index];
		}

		// Update the data pointers
		ownedData = std::move(newData);
		dataPtr = ownedData.data();
	}
	else
	{
		// We don't own the data, cannot rearrange it
		// Update only dimension metadata; data remains the same
		// However, this means accessing elements will produce incorrect results
		// It is safer to throw an exception
		throw std::runtime_error("Cannot swap dimensions on a non-owning tensor.");
	}

	std::cout << "Finished performing transpose" << std::endl;
}
