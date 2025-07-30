function cropSpikeSorterInput(chanMapFile, pathToBinFile, binFile, minFR, fs, k, windowLength, binLength, NChanTot, start, nMinutes, createTestData)
% Make folder for spike sorter
dirName = append('oss', num2str(k));
dir=fullfile(pathToBinFile, dirName);
inputDir=fullfile(pathToBinFile, fullfile(dirName, 'input'));
outputDir=fullfile(pathToBinFile, fullfile(dirName, 'output'));
mkdir(dir);
mkdir(inputDir);
mkdir(outputDir);

pathToKSOutput = fullfile(pathToBinFile, 'kilosort_output');
templates=readNPY(fullfile(pathToKSOutput,'templates.npy')); % (nTemplates x nSamples x nChannels)

% Make a mask that selects the k-most informative neuron templates
bestKMask = getBestKMask(pathToBinFile, pathToKSOutput, minFR, fs, k, windowLength, binLength);

% Crop templates along the template dimension 
templates = templates(bestKMask, :, :);

% Find out which channels are nonzero for the top k templates
templatesSum = squeeze(sum(templates, 1));
channelMask = sum(templatesSum) ~= 0; % TODO a small nonzero number may improve performance?

% Crop templates along the channel dimension and write to file
templates = templates(:, :, channelMask);
templatesFName = fullfile(inputDir, 'templates.npy');
writeNPY(templates, templatesFName);

% Crop whitening_mat and write to file
whiteningMat = readNPY(fullfile(pathToKSOutput, "whitening_mat.npy"));
whiteningMat = whiteningMat(channelMask, channelMask);
writeNPY(whiteningMat, fullfile(inputDir, 'whiteningMat.npy'))

% Crop Channel Map and write to file
load(chanMapFile, 'chanMap0ind', 'connected');
chanMap = chanMap0ind(connected);
chanMap = int32(chanMap);
chanMap = chanMap(channelMask);
channelMapFName = fullfile(inputDir, 'channelMap.npy'); % Different from channel_map saved by kilosort (which is used for phy)
writeNPY(chanMap, channelMapFName)

finish = fs * 60 * nMinutes;
cropBin(pathToBinFile, binFile, inputDir, start, finish, NChanTot, channelMask, createTestData)

end