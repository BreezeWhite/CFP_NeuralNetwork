function imdb = CFP_Extraction()

%%List files in the given dataset
wav_path = 'K:/Dataset/Wav_Files/';
gt_path = 'K:/Dataset/Ground_Truth/';
save_path = 'K:/Dataset/Extracted_Feature';

[wavList, midiList] = ReadFold(wav_path, gt_path);

%%Extraction parameters
start_t = 0; 
fr = 2.5; % frequency resolution, you may set it larger (but smaller than 4) if it takes too much memory
Hop = 441; % 0.01 sec
h = blackmanharris(7939); % 0.18 sec
fc = 20; % the frequency of the lowest pitch
tc = 1/4000; % the period of the highest pitch
NumPerOctave = 36;
use_channel = logical([1 1 1]);


sample_num_per_sec = 44100/Hop;
type_name = ['Spec'; 'Ceps'; 'GCoS'];
numSongs = size(wavList,1);
power = [0.24 0.6 1];
g = power(use_channel);
%g = [0.24 0.6 1]; %tfrLF = GCoS, tfrLQ = Cepstrum
%g = [0.24 0.6]; %tfrLF = Spectrum, tfrLQ = Cepstrum
%g = 0.24; %tfrLF = Spectrum, tfrLQ = 0

for i=1:numSongs
    %% Preprocess song information
    [x, fs]=audioread(wavList{i}); % Please try it using your own sample
    
    length_of_song = floor(size(x,1)/fs*sample_num_per_sec); % In samples. 0.01s/sample
    %length_of_song = ceil(size(x,1)/fs*sample_num_per_sec); % If midi length not equal to sample length, try use this one.
    fprintf('%s\n', wavList{i});
    fprintf('Song length: %d samples\n', length_of_song);
    
    %% Process feature
    MAX_Sample_num = 18000; % RAM peak usage: ~12GB
    if length_of_song > MAX_Sample_num % Process huge file part by part
        fs_range = MAX_Sample_num*Hop;
        round = ceil(length_of_song/MAX_Sample_num);
        tmpLF = cell(1,round);
        tmpLQ = cell(1,round);
        for ii = 0:(round-1)
            if ii == round-1
                tmpX = x((ii*fs_range+1):size(x,1));
            else
                tmpX = x(ii*fs_range+1:(ii+1)*fs_range);
            end
            [tfrLF, tfrLQ, t, central_frequencies] = CFP_filterbank(tmpX, fr, fs, Hop, h, fc, tc, g, NumPerOctave);
            tmpLF{ii+1} = tfrLF;
            tmpLQ{ii+1} = tfrLQ;
        end
        tfrLF = tmpLF{1};
        tfrLQ = tmpLQ{1};
        for ii = 2:length(tmpLF)
            tfrLF = cat(2,tfrLF,tmpLF{ii});
            tfrLQ = cat(2,tfrLQ,tmpLQ{ii});
        end
    else
        [tfrLF, tfrLQ, t, central_frequencies] = CFP_filterbank(x, fr, fs, Hop, h, fc, tc, g, NumPerOctave);
    end
    
    
    %% Load Ground Truth Data
    time_pt = (start_t+Hop/44100):Hop/44100:length_of_song/sample_num_per_sec;
    %label = ExtractMidiFromMid(strcat(song,'.mid'),time_pt,length_of_song); % Currently not working because the midi files are not precise.
    label = ExtractMidiFromTxt(midiList{i},time_pt); % For MAPS dataset
    
    %% Data check
    % If the length between extracted features and ground truth are note the same, the name of this song will be logged to file.  
    data = tfrLF;
    if size(data,2) ~= size(label,2)
        log = fopen('log.txt', 'a');
        fprintf(log, wavList{i});
        fprintf(log, '\n');
        fclose(log);
        fprintf('Error: %s - %d - %d',wavList{i}, size(data,2), size(label,2));
    end
    imdb.images.data = reshape(data, 1, 1, size(data,1), size(data,2));
    imdb.images.labels = reshape(label,1,1,size(label,1),size(label,2));
    
    %% Save to file
    name = strrep(wavList{i},'../MapsDataset/','');
    name = strrep(strrep(name,'/','_'),'.wav','');
    
    type_ = type_name(1,:);
    if length(g) == 3
        type_ = type_name(3,:);
    end
    maps = sprintf('%s/%s/%s.mat', save_path, type_, name); 
    
    save(maps,'imdb','-v7.3');
    
    if length(g)>1
        data = tfrLQ;
        imdb.images.data = reshape(data, 1, 1, size(data,1), size(data,2));
        maps = sprintf('%s/%s/%s.mat', save_path, type_name(2,:), name); 
        save(maps,'imdb','-v7.3');
    end
    
    i
end
end

%%
function [wavList, midiList] = ReadFold(wav_path, midi_path) 

wavlist = dir(wav_path);
wavlist = wavlist(3:end);
midilist = dir(midi_path);
midilist = midilist(3:end);

length = size(wavlist, 1);
wavList = cell(uint16(length), 1);
midiList = cell(uint16(length), 1);
for i = 1:length
    wavList{i} = [wav_path  wavlist(i).name];
    midiList{i} = [midi_path  midilist(i).name];
end
end


%%
function lab = ExtractMidiFromMid(fileName,time_pt,sample_num_per_song)

[~, n] = midi2nmat(fileName);
lab = -ones(88,sample_num_per_song);
for ti = 1:length(time_pt)
    for li = 1:size(n,1)
        if time_pt(ti) >= n(li,5) && time_pt(ti) <= n(li,6)
            lab(n(li,3)-20, ti) = 1;
        end
    end
end
end

%%
function lab = ExtractMidiFromTxt(fileName,time_pt)

fileName = strrep(fileName, '.mid', '.txt');
fileID = fopen(fileName,'r');
fgetl(fileID);
line = fscanf(fileID,'%f %f %f');
sz = numel(line);
n = reshape(line,3,sz/3);
n = n';
fclose(fileID);

lab = -ones(88,length(time_pt));
 for ti = 1:length(time_pt)
    for li = 1:size(n,1)
        if time_pt(ti) >= n(li,1) && time_pt(ti) <= n(li,2)
            lab(n(li,3)-20, ti) = 1;
        end
    end
end
end




