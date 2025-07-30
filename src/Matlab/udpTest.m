function udpTest
    % Create a datagram udp socket and send connectMsg to main server
    u = udpport("datagram", "IPV4");
    write(u, 5, "int32", "127.0.0.1", 8888)
    
    % Receive connectMsg from decoder's stimulusDisplayLiason
    connectMsg = read(u, 1, "int32");

	% Extract decoder's address and port information
    senderAddr = connectMsg.SenderAddress;
    senderPort = connectMsg.SenderPort;

	% Save the socket port number
    port = u.LocalPort;

	% Delete the datagram udp socket and reconstruct as byte-type udp socket
    clear u
    u = udpport("IPV4", 'LocalPort', port);
    
    
    % Now, in your loop
    % Get label somehow
    label = int32(1);
    
	% Send labels with this line
	write(u, label, "int32", senderAddr, senderPort);
end