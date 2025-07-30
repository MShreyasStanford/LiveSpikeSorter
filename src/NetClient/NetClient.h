#ifndef NETCLIENT_H
#define NETCLIENT_H

/* ---------------------------------------------------------------- */
/* Includes ------------------------------------------------------- */
/* ---------------------------------------------------------------- */

#include "Socket.h"

#include <vector>

/* ---------------------------------------------------------------- */
/* NetClient ------------------------------------------------------ */
/* ---------------------------------------------------------------- */

class NetClient : public Socket
{
private:
    std::vector<char>    vbuf;   // response buffer
    uint            	read_timeout_secs;

public:
    NetClient(
        const std::string  &host = "localhost",
        uint16          	port = 0,
        uint            	read_timeout_secs = 10 );
    virtual ~NetClient() {}

    virtual uint sendData(
        const void  *src,
        uint        srcBytes ) throw( const SockErr& );

    virtual uint receiveData(
        void    *dst,
        uint    dstBytes ) throw( const SockErr& );

    uint sendString( const std::string &s ) throw( const SockErr& );

    void rcvLine( std::vector<char> &line ) throw( const SockErr& );
    void rcvLines( std::vector<std::vector<char> > &vlines ) throw( const SockErr& );
};

#endif  // NETCLIENT_H


