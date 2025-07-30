#ifndef FRAGMENT_MANAGER_H
#define FRAGMENT_MANAGER_H

#include "Sock.h"
#include <thread>
#include <queue>
#include <mutex>
#include <iostream>
#include <condition_variable>
#include <map>
#include <chrono>
#include <atomic>
#include <tuple>

struct sockaddr_in;

// TODO: Lower the size of header without code breaking
#pragma pack(push, 1)
struct UDPFragmentHeader {
	bool isAck;
	uint32_t messageId;
	uint8_t fragmentIndex;
	uint8_t totalFragments;
	uint32_t fragmentSize;
};
#pragma pack(pop)

struct Datagram {
	std::unique_ptr<char[]> data;
	uint size = 0; // in bytes // timestamp of when datagram was last sent

	Datagram(size_t dataSize) : data(std::make_unique<char[]>(dataSize))
	{
		if (!data) throw std::runtime_error("Datagram::Datagram(): memory allocation error.\n");
	}
};

struct Fragment {
	UDPFragmentHeader header;
	std::unique_ptr<char[]> data;
	struct sockaddr_in dst;
	std::chrono::steady_clock::time_point sendTime;
	uint retries = 0;

	Fragment(const UDPFragmentHeader &h, const char *src, uint size) : 
		header(h), 
		data(std::make_unique<char[]>(size)), 
		sendTime(std::chrono::steady_clock::now()) 
	{
		if (!data) {
			throw std::runtime_error("Fragment::Fragment(): memory allocation error.\n");
		}

		memcpy(data.get(), src, size);
	}
};

struct KeyHash {
	std::size_t operator()(const std::tuple<uint32_t, uint8_t>& key) const {
		size_t hash1 = std::hash<uint32_t>{}(std::get<0>(key));
		size_t hash2 = std::hash<uint8_t>{}(std::get<1>(key));
		return hash1 ^ (hash2 << 1);
	}
};

class FragmentManager {
private:
	static const uint MAX_RETRIES = 3;	
	static const uint32_t FRAGMENT_SIZE = 60000; // TODO: Find "correct" value for this

	// the socket the class is managing alongside its mutex
	std::mutex sockMutex;
	Sock* sock;

	std::mutex assembledMutex;
	std::queue<std::unique_ptr<Datagram>> assembledDgs;
	std::condition_variable qCond;

	std::recursive_mutex unackMutex;
	std::unordered_map<
		std::tuple<uint64_t, uint8_t>, 
		std::unique_ptr<Fragment>, 
		KeyHash
	> unacknowledged;

	uint32_t generateUniqueMessageId();

public:
	FragmentManager(Sock *sock);
	~FragmentManager();

	void sendAck(uint32_t messageId, uint8_t fragmentIndex);
	uint send(
		const void  		*src,
		uint        		srcBytes,
		struct sockaddr_in	dstAddr = { 0 });

	uint recv(
		void    *dst,
		uint    dstBytes,
		int     clientFD = -1);

	uint recvSize();

	void assembler();
	void retransmitter();
	void networkMonitor();
	
	std::thread assemblerThread();
	std::thread retransmitterThread();
	std::thread networkMonitorThread();
};

#endif