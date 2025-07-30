function main_kilosort3(pathToKilosort, pathToNpyMatlab, chanMapFile, pathToBinFile, binFile, NchanTOT)
addpath(genpath(pathToKilosort)) % path to kilosort folder
addpath([pathToNpyMatlab]) % for converting to Phy

rootZ = [pathToBinFile]; % the raw data binary file is in this folder
rootH = pathToBinFile; % path to temporary binary file (same size as data, should be on fast SSD)

% Set the binary file 
ops.fbinary = fullfile(rootZ, binFile);

outputDir=fullfile(pathToBinFile, 'kilosort_output');

ops.trange    = [0 inf];  % time range to sort
ops.NchanTOT  = NchanTOT;

run('configFile384.m')
ops.fproc   = fullfile(rootH, 'temp_wh.dat'); % proc file on a fast SSD
ops.chanMap = chanMapFile;

%% this block runs all the steps of the algorithm
fprintf('Looking for data inside %s \n', rootZ)

% main parameter changes from Kilosort2 to v2.5
ops.sig        = 20;  % spatial smoothness constant for registration
ops.fshigh     = 300; % high-pass more aggresively
ops.nblocks    = 5; % blocks for registration. 0 turns it off, 1 does rigid registration. Replaces "datashift" option. 

% main parameter changes from Kilosort2.5 to v3.0
ops.Th       = [9 9];

% mp: avoid memory errors
ops.NT                  = 64*256+ ops.ntbuff; % must be multiple of 32 + ntbuff. This is the batch size (try decreasing if out of memory). 


rez                = preprocessDataSub(ops);
rez                = datashift2(rez, 1);

[rez, st3, tF]     = extract_spikes(rez);

rez                = template_learning(rez, tF, st3);

[rez, st3, tF]     = trackAndSort(rez);

rez                = final_clustering(rez, tF, st3);

rez                = find_merges(rez, 1);

mkdir(outputDir)
rezToPhy2(rez, outputDir);

%% 
