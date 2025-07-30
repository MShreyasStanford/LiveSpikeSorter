pathToMat = 'C:\SGL_DATA\01_27_p2_training_g2';
matName = '20230127_info.mat';
trainingNMinutes = 15;
%events 
negativeEventOnset=0;%300/1000;
sglx_timeshift=0;%3475539;

% --------------------------------------------

load(fullfile(pathToMat, matName), 'data'); % Don't need spatial freq / coordinate?



%which label to decode
data.label=data.stim_direction_1;
[counts,id]=groupcounts(data.stim_direction_1);
counts

trialspercondition=min(counts);
eventFileName = fullfile(pathToMat, 'eventFile_stimdirection_121.txt');
timing=data.stim_onset;

zerocount=0;
onecount=0;
fid = fopen(eventFileName, 'wb');
for i = 1:length(data.stim_onset) 
    if data.stim_onset(i) > trainingNMinutes * 60
        break
    end

    if round(data.label(i)) == 270 % CHECK THE STIM_DIRECTIONS
        label = 0;
        zerocount=zerocount+1;
    elseif round(data.label(i)) == 90
        label = 1;
        onecount=onecount+1;
    elseif round(data.label(i)) == 3
        label = 2;
    else % 4
        label = 3;
    end

    eventStreamSampleCt = round((timing(i)-negativeEventOnset) * 30000) + sglx_timeshift; % Convert seconds to streamSampleCounts
    if (label==0 && zerocount<=trialspercondition)
        fprintf(fid, '%d %d\n', eventStreamSampleCt, label);
    end

    if (label==1&&onecount<=trialspercondition)
        fprintf(fid, '%d %d\n', eventStreamSampleCt, label);
    end
end
fclose(fid);
%{
eventFileName = fullfile(pathToMat, 'testEventFile.txt');
fid = fopen(eventFileName, 'wb');
for j = i:length(data.stim_onset) 
    if round(data.stim_direction(j)) == 1 % CHECK THE STIM_DIRECTIONS
        label = 0;
    elseif round(data.stim_direction(j)) == 2
        label = 1;
    elseif round(data.stim_direction(j)) == 0
        label = 2;
    else % 180
        label = 3;
    end

    eventStreamSampleCt = round(data.stim_onset(j) * 30000) - trainingNMinutes * 60 * 30000;

    fprintf(fid, '%d %d\n', eventStreamSampleCt, label);
end
fclose(fid); 
%}