#ifndef NETWORKHELPERS_H
#define NETWORKHELPERS_H

#include "Sock.h"

// Index codes to indicate thread/computer's functionality
#define _SPIKE_SORTER_IMEC 0
#define _SPIKE_SORTER_NIDQ 1
#define _DECODER_IMEC 2
#define _DECODER_NIDQ 3
#define _GUI 4
#define _STIMULUS_DISPLAY 6


struct sockaddr_in getSetupAddr(const char* host, const uint16 port);
void sendConnectMsg(Sock *sock, struct sockaddr_in dstAddr, uint16 connectMsg);
sockaddr_in recvConnectMsg(Sock *sock, uint16 expected);

#endif