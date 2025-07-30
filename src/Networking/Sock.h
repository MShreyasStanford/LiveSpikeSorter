#ifndef SOCK_H
#define SOCK_H

/* ---------------------------------------------------------------- */
/* Includes-------------------------------------------------------- */
/* ---------------------------------------------------------------- */

#include <exception>
#include <string>

#if _WIN32 || _WIN64  // WINDOWS Includes

#include <winsock.h>
#include <io.h>

#else   // UNIX Includes

#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <errno.h>
#include <sys/select.h>
#include <sys/ioctl.h>
#include <unistd.h> // Basti added

#endif // WIN32 or UNIX

struct sockaddr_in;

/* ---------------------------------------------------------------- */
/* Types ---------------------------------------------------------- */
/* ---------------------------------------------------------------- */

typedef unsigned short  uint16;
typedef unsigned int    uint;
typedef unsigned long   ulong;

#if _WIN32 || _WIN64  // WINDOWS specific type

typedef int socklen_t;

#endif

/* ---------------------------------------------------------------- */
/* Classes -------------------------------------------------------- */
/* ---------------------------------------------------------------- */

/* ---------------------------------------------------------------- */
/* Class SockErr -------------------------------------------------- */
/* ---------------------------------------------------------------- */
/*
class SockErr : std::exception
{
protected:
	std::string msgid,
		reason;

public:
	SockErr(const std::string &reason = "") : reason(reason) {}
	virtual ~SockErr() throw() {}

	const std::string&  id() const throw() { return msgid; }
	const std::string& why() const throw() { return reason; }
	const char *what() const throw() { return reason.c_str(); }
};
*/
/* ---------------------------------------------------------------- */
/* Class ConnectionClosed ----------------------------------------- */
/* ---------------------------------------------------------------- */
/*
class ConnectionClosed : public SockErr
{
public:
	ConnectionClosed(
		const std::string &reason = "Connection closed by peer.")
		: SockErr(reason) {
		msgid = "CalinsNetMex:connectionClosed";
	}
};
*/
/* ---------------------------------------------------------------- */
/* Class HostNotFound --------------------------------------------- */
/* ---------------------------------------------------------------- */
/*
class HostNotFound : public SockErr
{
public:
	HostNotFound(
		const std::string &reason = "Host not found.")
		: SockErr(reason) {}
};
*/
/* ---------------------------------------------------------------- */
/* Class Socket --------------------------------------------------- */
/* ---------------------------------------------------------------- */

class Sock
{
private:
	struct sockaddr_in  m_addr;
	std::string         m_host,
		m_error;
	int                 m_sock,
		m_type,
		m_rcvBuf;
	uint16              m_port;
	bool                m_tcpNDelay,
		m_reuseAddr;

public:
	enum SocketType { TCP, UDP };
	enum SocketOption { TCPNoDelay, ReuseAddr };

	Sock(int type = TCP);
	virtual ~Sock();

	// Set/Get Functions
	void setHost(const std::string &host) { m_host = host; }
	const std::string &getHost() const { return m_host; }
	std::string errorReason() const { return m_error; }
	void setPort(uint16 port) { m_port = port; }
	uint16 getPort() const { return m_port; }
	struct sockaddr_in getAddr() const { return m_addr; }
	bool isValid() const { return m_sock > -1; }

	// Public Server Functions (eventually split into own classes!!)
	void setSocketOption(int option, bool enable);
	bool bind(uint16 port = 0);
	//int accept();
	int accept(struct sockaddr_in &clientAddr);

	// Public Client Functions
	bool connect(const std::string &host = "localhost", uint16 port = 0);
	void disconnect();

	// Common Functions
	uint sendData(
		const void  		*src,
		uint        		srcBytes,
		struct sockaddr_in	dstAddr = { 0 } // all fields 0
	); //throw(const SockErr&);

	uint recvData(
		void    *dst,
		uint    dstBytes,
		int     clientFD = -1);// throw(const SockErr&);

	uint bytesToRead();

	void setRcvBuf() const;
	void setSndBuf() const;

private:
	// Private Server Functions
	void setTCPNDelay() const;
	void setReuseAddr() const;

	// Private Client Functions
	void resolveHostAddr();
	static void resolveHostAddr(
		struct sockaddr_in  &addr,
		const std::string   &host,
		uint16              port);
};

#endif  // SOCK_H