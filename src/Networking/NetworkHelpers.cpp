#include <iostream>
#include "NetworkHelpers.h"
#include "../NetClient/Socket.h"

struct sockaddr_in getSetupAddr(const char* host, const uint16 port) {
	// Create zero'd out addr
	struct sockaddr_in addr;
	memset(&addr, 0, sizeof(addr));

	// Fill in family, host, and port info
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = inet_addr(host);
	addr.sin_port = htons(port);
	return addr;
}


void sendConnectMsg(Sock *sock, struct sockaddr_in dstAddr, uint16 connectMsg) {
	try {
		sock->sendData(&connectMsg, sizeof(int), dstAddr);
	}
	catch (const ConnectionClosed &e) {
		fprintf(stderr, "Connection closed. (%s)\n", e.why().c_str());
		exit(2);
	}
	catch (const SockErr &e) {
		fprintf(stderr, "Caught exception. (%s)\n", e.why().c_str());
		exit(1);
	}
}

sockaddr_in recvConnectMsg(Sock *sock, uint16 expected) {
	uint16 connectMsg;
	try {
		sock->recvData(&connectMsg, sizeof(uint16));
	}
	catch (const ConnectionClosed &e) {
		fprintf(stderr, "Connection closed. (%s)\n", e.why().c_str());
		exit(2);
	}
	catch (const SockErr &e) {
		fprintf(stderr, "Caught exception. (%s)\n", e.why().c_str());
		exit(1);
	}
	if (connectMsg != expected) {
		std::string errorMsg = "ERROR. Was expecting ";
		switch (expected) {
		case _DECODER_IMEC:
			errorMsg += "a decoder instance to connect";
			break;
		case _DECODER_NIDQ:
			errorMsg += "a decoder instance to connect";
			break;
		case _SPIKE_SORTER_IMEC:
			errorMsg += "a spike sorter instance to connect";
			break;
		case _SPIKE_SORTER_NIDQ:
			errorMsg += "a spike sorter instance to connect";
			break;
		case _GUI:
			errorMsg += "a GUI instance to connect";
			break;
		case _STIMULUS_DISPLAY:
			errorMsg += "a stimulus display instance to connect";
			break;
		}
		std::cout << errorMsg << std::endl;
		exit(-1);
	}
	struct sockaddr_in srcAddr = sock->getAddr();
	return srcAddr;
}