#include "SpikeStream.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>

#if _WIN32 || _WIN64
#ifndef NOMINMAX
#define NOMINMAX           // winsock.h pulls in windows.h, whose min/max macros break std::min
#endif
#include <winsock.h>
typedef SOCKET ss_socket_t;
typedef int socklen_t;
#define SS_CLOSE(s)        closesocket(static_cast<ss_socket_t>(s))
#define SS_INVALID         static_cast<uintptr_t>(INVALID_SOCKET)
#define SS_LASTERROR()     WSAGetLastError()
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cerrno>
typedef int ss_socket_t;
#define SS_CLOSE(s)        ::close(static_cast<ss_socket_t>(s))
#define SS_INVALID         static_cast<uintptr_t>(static_cast<ss_socket_t>(-1))
#define SS_LASTERROR()     errno
#endif

SpikeStream::SpikeStream()
	: m_enabled(false), m_sock(SS_INVALID), m_dstAddr(0), m_dstPort(0),
	m_sorterId(0), m_probe(0), m_batchSeq(0), m_sendErrors(0)
{
	m_buf.resize(sizeof(SpikeStreamHeader) + SPIKE_STREAM_MAX_RECORDS * sizeof(SpikeStreamRecord));
}

SpikeStream::~SpikeStream()
{
	if (m_sock != SS_INVALID)
		SS_CLOSE(m_sock);
#if _WIN32 || _WIN64
	if (m_enabled)
		WSACleanup();
#endif
}

bool SpikeStream::open(const std::string& hostPort, uint16_t sorterId, uint16_t probe)
{
	if (hostPort.empty())
		return false;

	const size_t colon = hostPort.rfind(':');
	if (colon == std::string::npos || colon == 0 || colon + 1 >= hostPort.size()) {
		std::cerr << "[SpikeStream] Expected host:port, got '" << hostPort << "'. Stream disabled." << std::endl;
		return false;
	}
	const std::string host = hostPort.substr(0, colon);
	int port = 0;
	try { port = std::stoi(hostPort.substr(colon + 1)); }
	catch (...) { port = 0; }
	if (port <= 0 || port > 65535) {
		std::cerr << "[SpikeStream] Bad port in '" << hostPort << "'. Stream disabled." << std::endl;
		return false;
	}

#if _WIN32 || _WIN64
	// Ref-counted; balanced by WSACleanup() in the destructor.
	WSADATA wsaData;
	if (WSAStartup(MAKEWORD(1, 1), &wsaData) != 0) {
		std::cerr << "[SpikeStream] WSAStartup failed. Stream disabled." << std::endl;
		return false;
	}
#endif

	uint32_t addr = inet_addr(host.c_str());
	if (addr == INADDR_NONE) {
		struct hostent* he = gethostbyname(host.c_str());
		if (!he || !he->h_addr) {
			std::cerr << "[SpikeStream] Could not resolve host '" << host << "'. Stream disabled." << std::endl;
#if _WIN32 || _WIN64
			WSACleanup();
#endif
			return false;
		}
		std::memcpy(&addr, he->h_addr, sizeof(addr));
	}

	// Unconnected socket + sendto: nothing is ever received on it, so an absent
	// listener (ICMP port unreachable) cannot poison a later call.
	m_sock = static_cast<uintptr_t>(::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP));
	if (m_sock == SS_INVALID) {
		std::cerr << "[SpikeStream] socket() failed (" << SS_LASTERROR() << "). Stream disabled." << std::endl;
#if _WIN32 || _WIN64
		WSACleanup();
#endif
		return false;
	}

	m_dstAddr = addr;
	m_dstPort = htons(static_cast<uint16_t>(port));
	m_sorterId = sorterId;
	m_probe = probe;
	m_enabled = true;
	std::cout << "[SpikeStream] Sorter " << sorterId << " (imec" << probe << ") streaming spikes to " << host << ":" << port << std::endl;
	return true;
}

void SpikeStream::sendBatch(const std::vector<long>& times,
                            const std::vector<long>& clusters,
                            const std::vector<float>& amplitudes,
                            uint64_t batchStartSample,
                            uint64_t batchEndSample,
                            uint32_t processTimeUs)
{
	if (!m_enabled)
		return;

	const size_t n = std::min(times.size(), std::min(clusters.size(), amplitudes.size()));
	const size_t nParts = std::max<size_t>(1, (n + SPIKE_STREAM_MAX_RECORDS - 1) / SPIKE_STREAM_MAX_RECORDS);

	SpikeStreamHeader hdr;
	std::memset(&hdr, 0, sizeof(hdr));
	hdr.magic = SPIKE_STREAM_MAGIC;
	hdr.version = SPIKE_STREAM_VERSION;
	hdr.sorterId = m_sorterId;
	hdr.probe = m_probe;
	hdr.batchSeq = m_batchSeq++;
	hdr.nParts = static_cast<uint16_t>(nParts);
	hdr.batchStartSample = batchStartSample;
	hdr.batchEndSample = batchEndSample;
	hdr.processTimeUs = processTimeUs;

	struct sockaddr_in dst;
	std::memset(&dst, 0, sizeof(dst));
	dst.sin_family = AF_INET;
	dst.sin_port = m_dstPort;
	dst.sin_addr.s_addr = m_dstAddr;

	for (size_t p = 0; p < nParts; ++p) {
		const size_t first = p * SPIKE_STREAM_MAX_RECORDS;
		const size_t count = (n > first) ? std::min<size_t>(SPIKE_STREAM_MAX_RECORDS, n - first) : 0;

		hdr.part = static_cast<uint16_t>(p);
		hdr.nRecords = static_cast<uint16_t>(count);
		hdr.sendTimeUs = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
			std::chrono::steady_clock::now().time_since_epoch()).count());

		char* out = m_buf.data();
		std::memcpy(out, &hdr, sizeof(hdr));
		out += sizeof(hdr);
		for (size_t i = first; i < first + count; ++i) {
			SpikeStreamRecord rec;
			rec.sample = static_cast<uint64_t>(times[i]);
			rec.cluster = static_cast<int32_t>(clusters[i]);
			rec.amplitude = amplitudes[i];
			std::memcpy(out, &rec, sizeof(rec));
			out += sizeof(rec);
		}

		const int bytes = static_cast<int>(out - m_buf.data());
		const int sent = static_cast<int>(::sendto(static_cast<ss_socket_t>(m_sock), m_buf.data(), bytes, 0,
		                          reinterpret_cast<const struct sockaddr*>(&dst), static_cast<socklen_t>(sizeof(dst))));
		if (sent != bytes) {
			// Never block or spam the hot path: report the first failure, then every 1000th.
			if (m_sendErrors % 1000 == 0)
				std::cerr << "[SpikeStream] sendto failed (" << SS_LASTERROR() << "), "
				          << (m_sendErrors + 1) << " failure(s) so far." << std::endl;
			++m_sendErrors;
		}
	}
}
