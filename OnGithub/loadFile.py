import h5py
import numpy as np

def LoadFile(fileName, loadLabel = True):
    maps = h5py.File(fileName,'r')
    data = maps['imdb']['images']['data'][:]
    label = None
    if loadLabel:
        label = maps['imdb']['images']['labels'][:]
        if label.shape[1] > 88:
            label = label[:,21:109]
        label = np.where(label.squeeze()>0, 1, 0)
    
    maps.close()
    
    return data, label
    
def LoadbyFold(fold_name, feature_path, use_type=['Spec', 'Ceps', 'GCoS'], t_range = None):
# Load songs according to the content of given the fold.
    f = open(fold_name, 'r')
    songs = f.readlines()
    f.close()
    
    Data = []
    Label = []
    Index = [] # Record each song's start index and length
    
    start_i = 0
    samples = 0
    for s_name in songs:
        print('Loading songs: %d/%d' % (songs.index(s_name)+1, len(songs)), end='\r')
        
        ### Load different types of feature of each song ### 
        tmp_d = []
        skipped = False
        for i in range(len(use_type)):
            s_name = s_name.replace('/', '_').replace('\n','').replace('.wav', '.mat')
            useType = use_type[i]
            data, label = LoadFile(feature_path+useType+'/'+s_name, loadLabel=True if i==0 else False) # only need to load labels once
            
            if i==0 and label.shape[0] != data.shape[0]:
                print('\nSkipped one song because sample number of label and data does not match')
                print('%s - Label shape: %s - Data shape: %s' % (s_name, str(label.shape), str(data.shape)))
                skipped = True
                break
            
            ### Dealing with the given range of using specific time slice ###
            if t_range != None:
                samples_sec = 100 # 0.01s/sample
                start_s = t_range[0]*samples_sec
                end_s = t_range[1]*samples_sec
                song_len = data.shape[0]
                
                if start_s >= song_len:
                    err_msg = 'The given start time longer than current song(%.2fs)' % (song_len/samples_sec)
                    print('Error messege: ' + err_msg + '\n')
                    assert False, err_msg
                if end_s > song_len:
                    end_s = song_len
                    if i == 0:
                        print('The given end time longer than current song(%.2fs). Resetting...' % (song_len/samples_sec))
                
                data = data[range(start_s, end_s)]
                if i==0:
                    label = label[range(start_s, end_s)]
            
            tmp_d.append(data)
            if i==0: 
                Label.append(label)
        if skipped: continue
        
        tmp_d = np.asarray(tmp_d)
        tmp_data = tmp_d[0]
        for i in range(1, len(use_type)):
            tmp_data = np.concatenate((tmp_data, tmp_d[i]), axis=3)
        shape = tmp_data.shape
        tmp_data = tmp_data.reshape((shape[0], shape[2], shape[1], shape[3]))
        Data.append(tmp_data)
        
        Index.append({'start_i': start_i, 'length': shape[0]})
        start_i += shape[0]
        samples += shape[0]
    
    shape = Data[0].shape
    data_shape = (samples, shape[1], shape[2], shape[3])
    
    return Data, Label, Index, data_shape

def Hack4LoadbyFold(numSongs, givenFold, outName, use_same = False):
# Create a new text file that is needed for LoadbyFold
# Random select $numSongs in $givenFold

    if use_same:
        old = open(outName, 'r')
        content = old.readlines()
        old.close()
        if len(content) < numSongs:
            print('The given number of songs is less than that in the given fold. Re-selecting...')
        elif len(content) > numSongs:
            print('Using same fold (%s)' % outName)
           
            f = open(outName, 'w')
            for i in range(numSongs):
                if '\n' not in content[i]:
                    content[i] += '\n'
                f.write(content[i])
            f.close()
            
            return outName
        else:
            print('Using same fold (%s)' % outName)
            return outName
    
    f = open(givenFold, 'r')
    songs = f.readlines()
    f.close() 
    songs = np.random.permutation(songs)
    
    f = open(outName, 'w')
    for i in range(numSongs):
        if '\n' not in songs[i]:
            songs[i] += '\n'
        f.write(songs[i])
    f.close()
    
    return outName

    
    
    
    
    
    
