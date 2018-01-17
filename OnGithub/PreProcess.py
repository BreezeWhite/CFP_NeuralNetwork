import numpy as np

def preDNN_processBatch(data, label, index, sel_idx):
# Process data for each batch    
    shape = (len(sel_idx), 1, data[0].shape[2], data[0].shape[3])
    new_data = np.zeros(shape)
    sel_label = np.zeros((shape[0], 88))
    
    cur_pos = 0
    for i in sel_idx:
        ith_song = 0
        # Find which song is this index lies
        for j in range(len(index)):
            start_i = index[j]['start_i']
            if (i >= start_i) and (i < start_i+index[j]['length']):
                ith_song = j
                break
        song_len = index[ith_song]['length']
        i -= index[ith_song]['start_i']
		
        x = data[ith_song][i]
		
        new_data[cur_pos] = x.reshape((1, 1, shape[2], shape[3]))
        sel_label[cur_pos] = label[ith_song][i]
        cur_pos += 1
		
    return new_data, sel_label
	
def preCNN_processBatch(data, label, index, sel_idx, numFrames):
# Process data for each batch
    assert numFrames%2 == 1    
    
    half_numFrames = int(numFrames/2)    
    shape = (len(sel_idx), numFrames, data[0].shape[2], data[0].shape[3])
    new_data = np.zeros(shape)
    sel_label = np.zeros((shape[0], 88))
    
    cur_pos = 0
    for i in sel_idx:
        ith_song = 0
        # Find which song is this index lies
        for j in range(len(index)):
            start_i = index[j]['start_i']
            if (i >= start_i) and (i < start_i+index[j]['length']):
                ith_song = j
                break
        song_len = index[ith_song]['length']
        i -= index[ith_song]['start_i']
        
        # Combine frames. For each sample, $half_numFrames frames before and after are combined together.
        # Here will check the border of each song. Zeros will be padded if the range is out of the bound.
        if i < half_numFrames:
            frame_idx = range(0, i+half_numFrames+1)
            x = np.zeros((numFrames, 1, shape[2], shape[3]))
            x[half_numFrames-i:] = data[ith_song][frame_idx]
        elif i >= song_len-half_numFrames:
            offset = song_len-i
            frame_idx = range(i-half_numFrames, i+offset)
            x = np.zeros((numFrames, 1, shape[2], shape[3]))
            x[0:half_numFrames+offset] = data[ith_song][frame_idx]
        else:
            frame_idx = range(i-half_numFrames, i+half_numFrames+1)
            x = data[ith_song][frame_idx]
            
        new_data[cur_pos] = x.reshape((1, numFrames, shape[2], shape[3]))
        sel_label[cur_pos] = label[ith_song][i]
        cur_pos += 1
		
    return new_data, sel_label
    
    
    
    
    
    
    
    
    
    
    
    
