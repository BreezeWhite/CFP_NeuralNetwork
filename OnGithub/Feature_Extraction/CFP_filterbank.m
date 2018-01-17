% output modes:
% 1) time-frequency (f-based and q-based)
% 2) time-log-frequency, with overlapped filterbank (f-based and q-based) 
% 3) semotone, max pooling and without filterbank (f-based and q-based)
function [tfrLF, tfrLQ, t, central_frequencies] = CFP_filterbank(x, fr, fs, Hop, h, fc, tc, g, NumPerOctave)

NumofLayer = length(g);

[tfr, f, t, N] = STFT(x, fr, fs, Hop, h);

tfr = abs(tfr.^g(1));

ceps = 0;
if NumofLayer >= 2
    for gc = 2:NumofLayer
        if rem(gc, 2)==0
            tc_idx = round(fs*tc);
            ceps = real(fft(tfr))./sqrt(N);
            ceps = nonlinear_func(ceps, g(gc), tc_idx);
        else
            fc_idx = round(fr/fc);
            tfr = real(fft(ceps))./sqrt(N);
            tfr = nonlinear_func(tfr, g(gc), fc_idx);
        end    
    end
end

tfr = tfr(1:round(N/2),:);
HighFreqIdx = round((1/tc)/fr)+1;
f = f(1:HighFreqIdx);
tfr = tfr(1:HighFreqIdx,:);
[tfrLF, central_frequencies.LF] = FreqToLogFreqMapping(tfr, f, fr, fc, tc, NumPerOctave);

tfrLQ = ceps;
if NumofLayer >= 2
    ceps = ceps(1:round(N/2),:);
    HighQuefIdx = round(fs/fc)+1;
    q = (0:HighQuefIdx-1)./fs;
    ceps = ceps(1:HighQuefIdx,:);
    [tfrLQ, central_frequencies.LQ] = QuefToLogFreqMapping(ceps, q, fs, fc, tc, NumPerOctave);
end
end

function [tfr, f, t, N] = STFT(x, fr, fs, Hop, h)
% STFT by Li Su, 2017
% Reference:
% Time-frequency toolbox by Patrick Flanderin
% fr: desired frequency resolution
% fs: sampling frequency
% Note: fr = alpha * fs ->
% alpha = fr / fs
% tfr: optput STFT (full frequency)
% f: output frequency grid (only positive frequency) 
% t: output time grid (from the start to the end)

if size(h,2) > size(h,1)
    h = h';
end

	% for tfr
alpha = fr/fs;
N = length(-0.5+alpha:alpha:0.5);
Win_length = max(size(h));
f = fs.*linspace(0, 0.5, round(N/2))' ;

Lh = floor((Win_length-1)/2);
t = Hop:Hop:floor(length(x)/Hop)*Hop;
x_Frame = zeros(N, length(t));
for ii = 1:length(t)
    ti = t(ii); 
    tau = -min([round(N/2)-1,Lh,ti-1]):min([round(N/2)-1,Lh,length(x)-ti]);
    indices= rem(N+tau,N)+1;
    norm_h=norm(h(Lh+1+tau)); 
    
    A = x(ti+tau)-mean(x(ti+tau));
    B = conj( h(Lh+1+tau));
    
    if size(A, 1) ~= size(B, 1)
        x_Frame(indices,ii) = A .* B' / norm_h;
    else
        x_Frame(indices,ii) = A .* B / norm_h;
    end
end

tfr = fft(x_Frame, N, 1);
end

function X = nonlinear_func(X, g, cutoff)
if g~=0
    X(X<0) = 0;
    X(1:cutoff, :) = 0;
    X(end-cutoff+1:end, :) = 0;
    X = X.^g;
else
    X = log(X);
    X(1:cutoff, :) = 0;
    X(end-cutoff+1:end, :) = 0;
end
end

function [tfrL, central_frequencies] = FreqToLogFreqMapping(tfr, f, fr, fc, tc, NumPerOctave)
StartFreq = fc; StopFreq = 1/tc;
Nest = ceil(log2(StopFreq/StartFreq))*NumPerOctave;

central_frequencies = [];
for i = 1:Nest
    CenFreq = StartFreq*2^((i-1)/NumPerOctave);
    if CenFreq < StopFreq
        central_frequencies = [central_frequencies; CenFreq];
    else
        break;
    end
end

Nest = length(central_frequencies);
freq_band_transformation = zeros(Nest-1, length(f));
for i = 2:length(central_frequencies)-1
    for j = round(central_frequencies(i-1)/fr+1) : round(central_frequencies(i+1)/fr+1)
        if (f(j)>central_frequencies(i-1) && f(j)<central_frequencies(i))
            freq_band_transformation(i,j) = (f(j)-central_frequencies(i-1))/(central_frequencies(i)-central_frequencies(i-1));%/D;
        elseif (f(j)>central_frequencies(i) && f(j)<central_frequencies(i+1))
            freq_band_transformation(i,j) = (central_frequencies(i+1)-f(j))/(central_frequencies(i+1)-central_frequencies(i));%/D;
        end
    end
end

tfrL = freq_band_transformation*tfr;
end

function [tfrL, central_frequencies] = QuefToLogFreqMapping(ceps, q, fs, fc, tc, NumPerOctave)
StartFreq = fc; StopFreq = 1/tc;
Nest = ceil(log2(StopFreq/StartFreq))*NumPerOctave;

central_frequencies = [];
for i = 1:Nest
    CenFreq = StartFreq*2^((i-1)/NumPerOctave);
    if CenFreq < StopFreq
        central_frequencies = [central_frequencies; CenFreq];
    else
        break;
    end
end

f = 1./q;
Nest = length(central_frequencies);
freq_band_transformation = zeros(Nest-1, length(q));
for i = 2:length(central_frequencies)-1
    for j = round(fs/central_frequencies(i-1)+1) :-1: round(fs/central_frequencies(i+1)+1)
        if (f(j)>central_frequencies(i-1) && f(j)<central_frequencies(i))
            freq_band_transformation(i,j) = (f(j)-central_frequencies(i-1))/(central_frequencies(i)-central_frequencies(i-1));%/D;
        elseif (f(j)>central_frequencies(i) && f(j)<central_frequencies(i+1))
            freq_band_transformation(i,j) = (central_frequencies(i+1)-f(j))/(central_frequencies(i+1)-central_frequencies(i));%/D;
        end
    end
end

tfrL = freq_band_transformation*ceps;
end