% sdm_bincounts_receiver.m — TCP receiver for BinCounts SDM packets
% Start this BEFORE launching OnlineSpikes.exe
%
% Usage: set ip/port, then run. OnlineSpikes connects to this server.

ip   = "0.0.0.0";   % listen on all interfaces
port = 9999;         % match your --sdm_port value

server = tcpserver(ip, port, "Timeout", 60);
fprintf("Waiting for OnlineSpikes to connect on port %d...\n", port);

% Wait for connection
while ~server.Connected
    pause(0.1);
end
fprintf("Connected!\n");

% --- Read hello packet ---
% First 10 bytes: uint64 sampleCt (=0) + uint16 numChannels
hdr = read(server, 10, "uint8");
sampleCt0 = typecast(uint8(hdr(1:8)), 'uint64');
N = typecast(uint8(hdr(9:10)), 'uint16');
assert(sampleCt0 == 0, "Expected hello packet with sampleCt=0");

% Next 4*N bytes: int32 channel IDs
ch_bytes = read(server, 4*double(N), "uint8");
channels = typecast(uint8(ch_bytes), 'int32');
fprintf("Hello: %d channels = [%s]\n", N, num2str(channels'));

% --- Stream data packets ---
pktSize = 10 + 4*double(N);
while server.Connected
    if server.NumBytesAvailable >= pktSize
        raw = read(server, pktSize, "uint8");
        sampleCt = typecast(uint8(raw(1:8)), 'uint64');
        counts   = typecast(uint8(raw(11:end)), 'single');
        fprintf("t=%d  counts=[%s]\n", sampleCt, num2str(counts'));
    else
        pause(0.001);
    end
end
fprintf("Disconnected.\n");
