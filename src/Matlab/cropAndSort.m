function cropAndSort

% You will likely need to modify these parameters
pathToKilosort = 'C:\Users\Spike Sorter\source\Kilosort-main\Kilosort-main';
pathToNpyMatlab = 'C:\Users\Spike Sorter\source\npy-matlab-master\npy-matlab-master\npy-matlab';
chanMapFile = 'chanMap_20220819.mat'; % 'neuropixPhase3A_kilosortChanMap.mat'; 
pathToBinFile = 'C:\SGL_DATA\08_23_settling_g0\08_23_settling_g0_imec0'; % Libsvm fails if this is not single quotes!!
binFile = '08_23_settling_g0_t0.imec0.ap.bin';  
NChanTot = 385; 
minFR = 1; % Any neuron with firing rate lower than minFR is cropped
windowLength = 15000;
binLength = 1500;
k = 25;

% To split your binfile into separate train and test, bin files set 
% createTestData to be true. The variables below are only significant if
% createTestData == true
createTestData = true; 
nMinutes = 6; 
fs = 30000.0;
start = 1; % use to skip over some of the beginning of bin file


% -------------------------------------------------------

main_kilosort3(pathToKilosort, pathToNpyMatlab, chanMapFile, pathToBinFile, binFile, NChanTot);
%{
cropSpikeSorterInput(chanMapFile, pathToBinFile, binFile, minFR, fs, k, windowLength, binLength, NChanTot, start, nMinutes, createTestData);
%}

end