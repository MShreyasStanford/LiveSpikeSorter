% Implementation is essentially identical to DataBinner.cpp/DataBinner.h
% TODO Learn to use c++ classes in Matlab to reduce code redundancy
classdef DataBinner < handle
    properties
        % For writing output
        fidOutput

        % Calculation variables
        windowLength {mustBeNumeric}
        binLength {mustBeNumeric}
        nBins {mustBeNumeric}
        nextBinTime {mustBeNumeric}
        currentIndex {mustBeNumeric}
        
        % Cells of maps
        bins
    end

    methods
        function obj = DataBinner(outputFileName, windowLength,binLength)
            obj.fidOutput = fopen(outputFileName, 'w');

            obj.windowLength = windowLength;
            obj.binLength = binLength;
            obj.nBins = windowLength / binLength;
            obj.nextBinTime = binLength;
            obj.currentIndex = 1;

            % Initialize bins with empty maps
            create_containers = @(n)arrayfun(@(x)containers.Map('KeyType', 'uint32', 'ValueType', 'double'), 1:n, 'UniformOutput', false);
            obj.bins = create_containers(obj.nBins);
        end


        function insert(obj, times, channels)
            % Iterate over the spikes
            for i = 1 : size(times, 2)
                % If data belongs to next bin
                if times(i) > obj.nextBinTime
                    % Set index to oldest bin
                    obj.currentIndex = mod((obj.currentIndex), obj.nBins) + 1;
                    
                    % Clear oldest bin for new data
                    remove(obj.bins{obj.currentIndex}, keys(obj.bins{obj.currentIndex}));

                    % Update next subbin time
                    obj.nextBinTime = obj.nextBinTime + obj.binLength;
                end

                % Increment spike count of the channel by 1
                incrementMap(obj.bins{obj.currentIndex}, channels(i), 1);
            end
        end

        function window = getDataWindow(obj)
            % Set window as empty map
            window = containers.Map('KeyType', 'uint32', 'ValueType', 'double');
            
            % Iterate over bins
            for i = 1 : obj.nBins
                % Iterate over bin's channel counts
                for channel = keys(obj.bins{i})
                    count = obj.bins{i}(channel{1});
                    incrementMap(window, channel{1}, count);
                end
            end
        end

        function writeDecoderInput(obj, label, window)
            fprintf(obj.fidOutput, '+%d', label);
            for channel = keys(window)
                count = window(channel{1});
                fprintf(obj.fidOutput, ' %d:%f', channel{1} + 1, count); % Channel +1 because libSVM is 1-indexed
            end
            fprintf(obj.fidOutput, '\n');
        end

        function readInSpikes(obj, spikeTimes, spikeTemplates, eventFileName)
            fid = fopen(eventFileName);
            events = fscanf(fid, '%d %d', [2, inf]);
            fclose(fid);

            eventTimes = events(1, :);
            eventLabels = events(2, :);

            times = [];
            channels = [];
            eventIdx = 1;
            
            for i = 1:size(spikeTimes, 1)
                % If a window's length of time has passed since the time
                % vent, record dataWindow and extract a new label
                if eventTimes(eventIdx) + obj.windowLength < spikeTimes(i)
                    % Insert Times and channels vectors into the binner
                    insert(obj, times, channels);
                    times = [];
                    channels = [];

                    window = getDataWindow(obj);

                    writeDecoderInput(obj, eventLabels(eventIdx), window);
                    
                    % Set eventIdx for the next event and its label
                    eventIdx = eventIdx + 1;

                    if eventIdx > size(eventTimes, 2)
                        break
                    end
                end

                % Insert time and channel
                times = [times spikeTimes(i)];
                channels = [channels spikeTemplates(i)];
            end

            fclose(obj.fidOutput);
        end
    end
end

% Helper function
function incrementMap(map, key, add)
    if isKey(map, key) % If key exists in map
        map(key) = map(key) + add;
    else % If key doesn't exist in map
        map(key) = add;
    end
end