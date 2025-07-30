#include "FragmentManager.h"
#include <sstream>
#include <unordered_map>
#include "../Helpers/Timer.h"

static std::atomic<uint32_t> sequence_number(0);
static std::atomic<long> dgsSent = 0;
static std::atomic<long> dgsRcvd = 0;
static std::atomic<long> packetsLost = 0;
static std::atomic<long> retransmissions = 0;
static std::atomic<long long> bytesRcvd = 0;

FragmentManager::FragmentManager(Sock *sock)
	: sock(sock)
{

}

FragmentManager::~FragmentManager()
{
	{
		std::lock_guard<std::mutex> lock(assembledMutex);
		while (!assembledDgs.empty()) assembledDgs.pop();
	}
}

void FragmentManager::sendAck(
	uint32_t messageId,
	uint8_t fragmentIndex
)
{
	char buf[sizeof(UDPFragmentHeader)];
	memset(buf, 0, sizeof(UDPFragmentHeader));
	UDPFragmentHeader* header = reinterpret_cast<UDPFragmentHeader*>(buf);
	header->isAck = true;
	header->messageId = messageId;
	header->fragmentIndex = fragmentIndex;
	header->totalFragments = 1;
	header->fragmentSize = 0;

	std::lock_guard<std::mutex> lock(sockMutex);
	if (sock->sendData(buf, sizeof(UDPFragmentHeader), sock->getAddr()) == SOCKET_ERROR) {
		std::cerr << "ack() error: sendData() error code " << WSAGetLastError() << "\n";
	}
}


uint FragmentManager::send(
	const void  		*src,
	uint        		srcBytes,
	struct sockaddr_in	dstAddr)
{
	if (!srcBytes)
		return 0;

	uint32_t messageId = generateUniqueMessageId();
	uint totalBytesSent = 0;
	char buf[FRAGMENT_SIZE + sizeof(UDPFragmentHeader)];
	uint8_t totalFragments = (srcBytes + FRAGMENT_SIZE - 1) / FRAGMENT_SIZE;

	for (uint8_t i = 0; i < totalFragments; i++) {
		uint32_t fragmentSize = min(FRAGMENT_SIZE, srcBytes - i * FRAGMENT_SIZE);
		memset(buf, 0, FRAGMENT_SIZE + sizeof(UDPFragmentHeader));
		UDPFragmentHeader* header = reinterpret_cast<UDPFragmentHeader*>(buf);
		header->isAck = false;
		header->messageId = messageId;
		header->fragmentIndex = i;
		header->totalFragments = totalFragments;
		header->fragmentSize = fragmentSize;
		memcpy(buf + sizeof(UDPFragmentHeader), (char*)src + i * FRAGMENT_SIZE, fragmentSize);
		std::lock_guard<std::recursive_mutex> lock(unackMutex);
		{
			std::lock_guard<std::mutex> lock(sockMutex);
			totalBytesSent += sock->sendData(buf, sizeof(UDPFragmentHeader) + fragmentSize, dstAddr) - sizeof(UDPFragmentHeader);
		}
		auto key = std::make_tuple(messageId, i);
		if (unacknowledged.find(key) == unacknowledged.end()) {
			unacknowledged[key] = std::make_unique<Fragment>(Fragment(*header, buf + sizeof(UDPFragmentHeader), fragmentSize));
			if (!unacknowledged[key]) {
				throw std::runtime_error("Failed to store unacknowledged packet.\n");
			}
			unacknowledged[key]->dst = dstAddr;
		}
	}

	if (totalBytesSent < srcBytes) {
		std::ostringstream oss;
		oss << "send() error: sent " << totalBytesSent << " bytes, expected to send " << srcBytes << " bytes\n";
		throw std::runtime_error(oss.str());
	}

	dgsSent++;
	return totalBytesSent;
}

uint FragmentManager::recv(
	void    *dst,
	uint    dstBytes,
	int     clientFD)// throw(const SockErr&)
{
	if (!dstBytes)
		return 0;

	std::unique_lock<std::mutex> lock(assembledMutex);

	// wait until a datagram is fully assembled
	qCond.wait(lock, [this] { return !assembledDgs.empty(); });

	// copy to dst, cleanup, and return
	auto& dg = assembledDgs.front();
	uint size = dg->size;
	memcpy(dst, dg->data.get(), size);
	assembledDgs.pop();

	// debug logs
	dgsRcvd++;
	bytesRcvd += size;

	return size;
}

uint FragmentManager::recvSize() {
	std::unique_lock<std::mutex> lock(assembledMutex);
	qCond.wait(lock, [this] { return !assembledDgs.empty(); });
	return assembledDgs.front()->size;
}

uint32_t FragmentManager::generateUniqueMessageId() {
	return sequence_number.fetch_add(1, std::memory_order_relaxed);
}

void FragmentManager::assembler() {
	char buf[FRAGMENT_SIZE + sizeof(UDPFragmentHeader)];
	memset(buf, 0, FRAGMENT_SIZE + sizeof(UDPFragmentHeader));
	std::unordered_map<uint32_t, std::unique_ptr<Datagram>> idToDatagram;
	std::unordered_map<uint32_t, uint> nFragsRcvd;

	while (true) {
		// this is needed because otherwise the next block just hogs the mutex
		// however... the issue is sock->bytesToRead() is not atomic, and thus
		// could lead to race conditions in the sock, so...
		// TODO: get rid of race conditions in Sock
		if (!sock->bytesToRead()) continue;

		// read data from socket
		uint bytesRead;
		{
			std::lock_guard<std::mutex> lock(sockMutex);
			bytesRead = sock->recvData(
				static_cast<char*>(buf),
				FRAGMENT_SIZE + sizeof(UDPFragmentHeader)
			);
		}

		// if winsock throws error
		if (bytesRead == SOCKET_ERROR || bytesRead == 0) {
			//std::cerr << "recvFrom() error code " << WSAGetLastError() << "\n";
			continue;
		}

		// parse header
		UDPFragmentHeader *header = reinterpret_cast<UDPFragmentHeader*>(buf);
		bool isAck = header->isAck;
		uint32_t messageId = header->messageId;
		uint8_t fragmentIndex = header->fragmentIndex;

		// if recv'd ACK, remove datagram from list of unACK'd datagrams
		if (isAck) {
			std::lock_guard<std::recursive_mutex> lock(unackMutex);
			unacknowledged.erase(std::make_tuple(messageId, fragmentIndex));
			continue;
		}

		// parse rest of header
		uint8_t totalFragments = header->totalFragments;
		uint32_t fragmentSize = header->fragmentSize;

		// send ACK for fragment
		sendAck(messageId, fragmentIndex);

		if (bytesRead != fragmentSize + sizeof(UDPFragmentHeader)) {
			throw std::runtime_error("recv() failed: number of bytes read does not match expected.");
		}

		// if bad information is received, prevents segfault
		if (fragmentIndex >= totalFragments || fragmentSize > FRAGMENT_SIZE) {
			fprintf(stderr, "receiveAndAssemble() error: incorrect header information received.");
			continue;
		}

		// create entry for datagram if it doesn't exist
		if (idToDatagram.find(messageId) == idToDatagram.end()) {
			idToDatagram[messageId] = std::make_unique<Datagram>(FRAGMENT_SIZE * totalFragments);
			nFragsRcvd[messageId] = 0;
			if (!idToDatagram[messageId]) {
				fprintf(stderr, "receiveAndAssemble() error: out of memory, datagram skipped.\n");
				continue;
			}
		}

		// copy received data into map
		memcpy(
			idToDatagram[messageId]->data.get() + FRAGMENT_SIZE * fragmentIndex, 
			buf + sizeof(UDPFragmentHeader), 
			fragmentSize
		);

		nFragsRcvd[messageId]++;
		idToDatagram[messageId]->size += fragmentSize;

		if (nFragsRcvd[messageId] == totalFragments) {
			std::lock_guard<std::mutex> lock(assembledMutex);

			// put assembled diagram into queue for retreival
			assembledDgs.push(std::move(idToDatagram[messageId]));
			
			// unlock wait in recv() so that it can access the queue
			qCond.notify_one();

			// erase entries corresponding to this datagram
			idToDatagram.erase(messageId);
			nFragsRcvd.erase(messageId);
		}	
	}
}

void FragmentManager::retransmitter() {
	const std::chrono::milliseconds timeout(2);
	while (true) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
		auto now = std::chrono::steady_clock::now();
		std::vector<std::tuple<uint32_t, uint8_t>> toRemove;

		{
			std::lock_guard<std::recursive_mutex> unackLock(unackMutex);

			for (auto &entry : unacknowledged) {
				if (std::chrono::duration_cast<std::chrono::milliseconds>(now - entry.second->sendTime) > timeout) {
					if (entry.second->retries >= MAX_RETRIES) {
						toRemove.push_back(entry.first); // Give up after MAX_RETRIES
						/*
						std::cerr << "Packet loss: packet #" << entry.second->header.messageId
							<< " index " << (int)entry.second->header.fragmentIndex
							<< "/" << (int)entry.second->header.totalFragments << std::endl;*/
						packetsLost++;
					}
					else {
						/*
						std::cout << "Retransmitting Packet #" << entry.second->header.messageId
							<< " index " << (int) entry.second->header.fragmentIndex 
							<< "/" << (int) entry.second->header.totalFragments << " ["
							<< entry.second->header.fragmentSize << " bytes] for the " << entry.second->retries << "th time\n";*/
							
						// retransmit fragment
						char buf[FRAGMENT_SIZE + sizeof(UDPFragmentHeader)];
						memset(buf, 0, FRAGMENT_SIZE + sizeof(UDPFragmentHeader));
						memcpy(buf, reinterpret_cast<char*>(&entry.second->header), sizeof(UDPFragmentHeader));
						memcpy(buf + sizeof(UDPFragmentHeader), entry.second->data.get(), entry.second->header.fragmentSize);
						{
							std::lock_guard<std::mutex> lock(sockMutex);
							sock->sendData(buf, sizeof(UDPFragmentHeader) + entry.second->header.fragmentSize, entry.second->dst);
						}
						retransmissions++;

						// update metadata needed for retransmission
						entry.second->sendTime = now; // Update send time
						entry.second->retries++; // Increment retry counter
					}
				}
			}
			for (auto id : toRemove) {
				unacknowledged.erase(id); // Remove packets that have expired
			}
		} // mutex unlocks here
	}
}

void FragmentManager::networkMonitor() {
	while (!sock->bytesToRead());
	auto startTime = std::chrono::steady_clock::now();

	while (true) {
		std::this_thread::sleep_for(std::chrono::seconds(5));
		auto now = std::chrono::steady_clock::now();
		std::stringstream ss;
		ss << "Duration: " << std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count() << "s\n"
			<< "Throughput: " << bytesRcvd / std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count() / 1000 << " MB/s\n"
			<< "Packets sent per second: " << dgsRcvd / std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count() << "\n"
			<< "Packets sent: " << dgsSent << "\n"
			<< "Packets received: " << dgsRcvd << "\n"
			<< "Packets lost: " << packetsLost << "\n"
			<< "Percentage of packets lost: " << (long double)packetsLost / (long double)dgsSent * 100 << "%\n"
			<< "Number of retransmissions: " << retransmissions << "\n"
			<< "Retransmissions per packet: " << (long double)retransmissions / (long double)dgsSent << std::endl;
		std::cout << ss.str();
	}

}
std::thread FragmentManager::assemblerThread() {
	return std::thread([this] { assembler(); });
}

std::thread FragmentManager::retransmitterThread() {
	return std::thread([this] { retransmitter();  });
}

std::thread FragmentManager::networkMonitorThread() {
	return std::thread([this] { networkMonitor(); });
}