function cropBinCalvin(pathToBinFile, binFile, folderName, outFile, streamSampleCtStart, streamSampleCtEnd, NChanTot)
    % Open original bin file accquired by spikeglx
    fidInput = fopen(fullfile(pathToBinFile, binFile), 'rb');
    
    % Data to be fed into onlinespikesorte
    trainBinFName = fullfile(folderName, outFile); % reconsider naming scheme
    fidOutput = fopen(trainBinFName, 'wb');
    
    % Iterates over scans from start of file to streamSampleCtEnd
    ntBuff = 64*256;
    for i = 1 : round(streamSampleCtEnd / ntBuff)
        [dataBatch, count] = fread(fidInput, [NChanTot ntBuff], '*int16'); 
        if count ~= NChanTot*ntBuff  %  reached end of file when count == 0, so break out
            warning('While cropping the data, the file reached eof at stream sample = %d.', i * ntBuff);
            fclose(fidInput);
            fclose(fidOutput);
            return
        end
        if i * ntBuff < streamSampleCtStart % Do not write to file any scans before streamSampleCtStart
            continue
        end
    
        fwrite(fidOutput, dataBatch, '*int16');
    end
    fclose(fidOutput);
end