import scipy
import numpy as np
import scipy.signal
import soundfile as sf

from CFP_filterbank import CFP_filterbank

def PreProcessSong(wav):
    
    Hop = 441
    h = scipy.signal.blackmanharris(7939)
    fr = 2.5
    fc = 20.0 # the frequency of the lowest pitch
    tc = 1/4000.0 # the period of the highest pitch
    g = np.array([0.24, 0.6, 1])
    NumPerOctave = 36
    MAX_Sample_Num = 5000
	
    x, fs = sf.read(wav)
    
    samples_of_song = np.floor(len(x)/Hop)
    print("Total %d samples to process." % int(samples_of_song))
    if samples_of_song > MAX_Sample_Num:
        freq_width = MAX_Sample_Num * Hop
        Round = int(np.ceil(samples_of_song/MAX_Sample_Num))
        tmpLF = []
        tmpLQ = []
        for i in range(Round):
            print("Round: %d/%d" % ((i+1), Round))
            if i == Round-1:
                tmpX = x[i*freq_width:]
            else:
                tmpX = x[i*freq_width:(i+1)*freq_width+1]
			
            tfrLF, tfrLQ, f, q, t, central_frequencies = CFP_filterbank(tmpX, fr, fs, Hop, h, fc, tc, g, NumPerOctave)
            tmpLF.append(tfrLF)
            tmpLQ.append(tfrLQ)
			
        tfrLF = tmpLF.pop(0)
        tfrLQ = tmpLQ.pop(0)
        for i in range(Round-1):
            tfrLF = np.concatenate((tfrLF, tmpLF.pop(0)), axis=1)
            tfrLQ = np.concatenate((tfrLQ, tmpLQ.pop(0)), axis=1)
    else:
        tfrLF, tfrLQ, f, q, t, central_frequencies = CFP_filterbank(x, fr, fs, Hop, h, fc, tc, g, NumPerOctave)

			
        
    tfrLF = tfrLF.transpose()
    tfrLQ = tfrLQ.transpose()
    tfrLF = tfrLF.reshape((tfrLF.shape[0],1,tfrLF.shape[1],1))
    tfrLQ = tfrLQ.reshape((tfrLQ.shape[0],1,tfrLQ.shape[1],1))        
    data = np.concatenate((tfrLQ, tfrLF), axis=3)
    
    return data