#ifndef SPIKESTREAM_H
#define SPIKESTREAM_H

// Public spike stream: sorted spikes out of LSS to any external process
// (Python, MATLAB, Bonsai, ...), so closed-loop logic can live outside the C++.
// Enabled with --spike_stream <host:port>; off by default.
//
// Spikes are exactly the rows written to spikeOutput.txt: de-duplicated, overlap
// trimmed, times in SpikeGLX imec stream samples, cluster = final-cluster index
// from closestCluster().
//
// Reference receiver: src/Python/lss_stream.py. Keep the two in lockstep.

#include <cstdint>
#include <string>
#include <vector>

static const uint32_t SPIKE_STREAM_MAGIC = 0x3153534C;   // "LSS1" as little-endian bytes
static const uint16_t SPIKE_STREAM_VERSION = 1;
static const int      SPIKE_STREAM_MAX_RECORDS = 89;

struct SpikeStreamHeader {
	uint32_t magic;             // SPIKE_STREAM_MAGIC
	uint16_t version;           // SPIKE_STREAM_VERSION
	uint16_t sorterId;          // GPU/sorter index
	uint32_t batchSeq;          // increments once per batch; gaps = dropped packets
	uint16_t part;              // 0 .. nParts-1
	uint16_t nParts;
	uint64_t batchStartSample;  // [start, end) = stream interval this batch's output covers;
	uint64_t batchEndSample;    // a gap to the next batch's start = skipped stream data
	uint64_t sendTimeUs;        // std::chrono::steady_clock (QueryPerformanceCounter on Windows)
	uint32_t processTimeUs;     // sorter compute time for this batch
	uint16_t nRecords;          // records in THIS datagram
	uint16_t probe;             // imec probe this sorter reads N = imecN (default was 0)
};
static_assert(sizeof(SpikeStreamHeader) == 48, "SpikeStreamHeader wire size changed");

struct SpikeStreamRecord {
	uint64_t sample;            // SpikeGLX imec stream sample
	int32_t  cluster;           // final-cluster index (templateMap.npy maps it to a Kilosort cluster ID)
	float    amplitude;
};
static_assert(sizeof(SpikeStreamRecord) == 16, "SpikeStreamRecord wire size changed");

class SpikeStream {
public:
	SpikeStream();
	~SpikeStream();

	// Returns false (and stays disabled) on a malformed or unresolvable address.
	bool open(const std::string& hostPort, uint16_t sorterId, uint16_t probe = 0);
	bool isEnabled() const { return m_enabled; }

	void sendBatch(const std::vector<long>& times,
	               const std::vector<long>& clusters,
	               const std::vector<float>& amplitudes,
	               uint64_t batchStartSample,
	               uint64_t batchEndSample,
	               uint32_t processTimeUs);

private:
	bool      m_enabled;
	uintptr_t m_sock;           // SOCKET on Windows, int fd elsewhere
	uint32_t  m_dstAddr;        // network byte order
	uint16_t  m_dstPort;        // network byte order
	uint16_t  m_sorterId;
	uint16_t  m_probe;
	uint32_t  m_batchSeq;
	uint64_t  m_sendErrors;
	std::vector<char> m_buf;
};

#endif
