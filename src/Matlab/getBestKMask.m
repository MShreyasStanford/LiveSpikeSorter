function bestKMask = getBestKMask(pathToBinFile, pathToKSOutput, minFR, fs, k, windowLength, binLength)
eventFileName = fullfile(pathToBinFile, 'eventFile.txt');
times=readNPY(fullfile(pathToKSOutput,'spike_times.npy'));
templates = readNPY(fullfile(pathToKSOutput,'spike_templates.npy'));
outputFileName = fullfile(pathToKSOutput, 'binnedSpikes.txt'); % Libsvm fails if this is not single quotes!!

% Create a mask that filters out muas/templates with low firing rates
mask = getMinFRAndGoodMask(pathToKSOutput, minFR, fs, times);

% Use the mask as the first step of filtering out less important templates
templatesRange = 1:size(mask);
filteredTemplatesMask = templatesRange(mask);
filteredTemplatesMask2 = ismember(templates, filteredTemplatesMask);
templates = templates(filteredTemplatesMask2);
times = times(filteredTemplatesMask2);

% Convert kilosort outputs times templates, and event times to libsvm format
binner = DataBinner(outputFileName, windowLength, binLength);
readInSpikes(binner, times, templates, eventFileName);
[label, data] = libsvmread(outputFileName);

% Find an optimal c value via cross validation
n = -13:13;
accuracy = nan(size(n));
for i=1:numel(n)
    c=2^n(i);       
    % perform 5-fold cross validation
    accuracy(i) = svmtrain(label, data,['-v 5 -q -t 0 -c ' num2str(c)]); 
end
[acc, i] = max(accuracy); 
disp(['Uncropped kilosort output Cross-validation accuracy: ', num2str(acc), '%'])
c = 2^n(i);      

% Using found c and uncropped spike data, train a model
model = svmtrain(label, data,['-q -t 0 -c ' num2str(c)]);

% Compute the model's weight matrix (possible because using linear kernel)
W = model.sv_coef' * full(model.SVs);

% Create a mask to select the k top neurons
[~, bestK ] = maxk(abs(W), k);
bestKMask = ismember((1:size(data,2))', bestK);

% Run to test if bestKMask is working/k is large neough
acc = svmtrain(label, data(:, bestKMask),['-v 5 -q -t 0 -c ' num2str(c)]); 
disp(['Cropped kilosort output Cross-validation accuracy: ', num2str(acc), '%'])
end

function mask = getMinFRAndGoodMask(pathToKSOutput, minFR, fs, times)
    KSLabels=tdfread(fullfile(pathToKSOutput,'cluster_KSLabel.tsv'));
    SpikeIdentity=readNPY(fullfile(pathToKSOutput,'spike_clusters.npy'));

    % Calculate firing rates and create mask for templates above minFR
    numTemplates=length(KSLabels.KSLabel);
    bins = 0:numTemplates;
    templateCounts = transpose(histcounts(SpikeIdentity, bins));
    SessionLength=(double(times(end))/fs);
    firingRates = templateCounts/SessionLength;
    firingRateMask = firingRates > minFR;

    % Create mask for good templates
    %goodMask = string(KSLabels.KSLabel) == "good";

    %mask = and(goodMask, firingRateMask);
    mask = firingRateMask;
end