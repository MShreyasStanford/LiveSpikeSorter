% sdm_receiver.m — Listen for SDM packets (13 bytes each)
% Usage: set port to match --sdm_port, then run
%
% Packet format (13 bytes):
%   Byte  0:     uint8  direction (1 = above threshold, 0 = below, 255 = below negative threshold)
%   Bytes 1-4:   float  value (z-score in zscore mode, P(class1) in logreg mode)
%   Bytes 5-12:  uint64 SpikeGLX sample count at bin end

port = 9999;  % match your --sdm_port value

u = udpport("byte", "LocalPort", port);
fprintf("Listening on UDP port %d...\n", port);

cleanupObj = onCleanup(@() delete(u));

while true
    if u.NumBytesAvailable >= 13
        raw = read(u, 13, "uint8");

        % Byte 0: direction (uint8, where 255 = -1)
        dirRaw = raw(1);
        if dirRaw == 255
            direction = -1;
        else
            direction = int8(dirRaw);  % 0 or 1
        end

        % Bytes 1-4: float value (z-score or P(class1))
        value = typecast(uint8(raw(2:5)), 'single');

        % Bytes 5-12: uint64 sample count
        sampleCt = typecast(uint8(raw(6:13)), 'uint64');

        fprintf("dir=%+d  value=%+8.4f  sampleCt=%d\n", direction, value, sampleCt);
    else
        pause(0.001);
    end
end
