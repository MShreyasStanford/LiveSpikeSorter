#ifndef SERIALIZATIONHELPERS_H
#define SERIALIZATIONHELPERS_H

#include <string>
#include "Sock.h"
#include "FragmentManager.h"
// Network Serialization and Deserialization Helpers

template <class T>
std::string serialize(T &payload) {
	std::stringstream ss;
	{ // Block necessary to ensure destructor of archive is called
		cereal::BinaryOutputArchive archive(ss);
		archive(payload);
	}

	return 	ss.str(); //Serialized version of the payload 
}

template <class T>
T deserialize(std::string &serializedPayload) {
	if (serializedPayload.empty()) {
		std::cout << "Deserializing an empty string.\n";
		throw std::runtime_error("Deserialization failed: Empty payload.");
	}

	T payload;
	std::stringstream ss;
	ss.write(serializedPayload.data(), serializedPayload.size());

	if (!ss.good()) {
		throw std::runtime_error("Failed to write serialized data to stringstream.");
	}
	try {
		cereal::BinaryInputArchive archive(ss);
		archive(payload);
	}
	catch (cereal::Exception &e) {
		std::cout << "Deserialization failed: " << e.what() << std::endl;
		std::cout << "Payload length during failure: " << serializedPayload.length() << std::endl;
		throw std::runtime_error("Deserialization failed: " + std::string(e.what()));
	}
	return payload;
}

template <class T>
void sendPayload(FragmentManager *fm, const T &payload, const sockaddr_in &addr) {
	std::string serializedPayload = serialize(payload);
	size_t payloadLength = serializedPayload.length();
	fm->send(serializedPayload.c_str(), payloadLength, addr);
}

template <class T>
T recvPayload(FragmentManager *fm) {
	size_t payloadLength = fm->recvSize();
	std::string serializedPayload;
	char* buf = (char*)malloc(payloadLength * sizeof(char));

	if (!fm) {
		throw std::runtime_error("recvPayload(): FragmentManager not found.\n");
	}

	if (!buf) {
		throw std::runtime_error("recvPayload(): malloc error\n");
	}

	if (uint bytesRcvd = fm->recv(buf, payloadLength) < payloadLength) {
		std::cerr << "recvPayload() expected " << payloadLength
			<< " received " << bytesRcvd << std::endl;
	}


	try
	{
		serializedPayload.assign(buf, payloadLength);
	}
	catch (const std::out_of_range& e) {
		std::cerr << "Out of range error: " << e.what() << std::endl;
	}
	catch (const std::length_error& e) {
		std::cerr << "Length error: " << e.what() << std::endl;
	}
	catch (const std::exception& e) {
		std::cerr << "General std::exception: " << e.what() << std::endl;
	}
	catch (...) {
		std::cerr << "An unexpected error occurred." << std::endl;
	}

	T payload = deserialize<T>(serializedPayload);
	free(buf);
	return payload;
}

#endif