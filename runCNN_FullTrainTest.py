import time
import h5py
import numpy as np
import Statistics as st
import matplotlib.pyplot as plt

from PreProcess import preCNN_processBatch
from loadFile import LoadbyFold, Hack4LoadbyFold

import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.utils.generic_utils import get_custom_objects


def runCNN_Train(feature_path, fold='config2_fold_1',useType=['Spec', 'Ceps', 'GCoS'], use_time_range = None):

    ############### Load and Preprocess Data ###############
    numVal = 30  #number of validation songs
    numTrain = 180  #number of training songs
    use_same_val = True  #weither to use the same validation set as last run
    use_same_train = True  #wether to use the same training set as last run
    numFrame = 5  #augment the time dimension of 5 (0.05s)
    
    val_fold = 'val_' + fold + '.txt'
    train_fold = 'train_' + fold + '.txt'
    #val_fold = Hack4LoadbyFold(numVal, 'test_%s.txt' % fold, 'val.txt', use_same=use_same_val) #you can specify your own fold
    #train_fold = Hack4LoadbyFold(numTrain, 'train_%s.txt' % fold, 'train.txt', use_same=use_same_train)
    
    print('Loading training data')
    train_data, train_label, train_index, tData_shape = LoadbyFold(train_fold, feature_path, use_type = useType, t_range = use_time_range)
    t_samples = tData_shape[0]
    
    print('\nLoading validation data')
    val_data, val_label, val_index, vData_shape = LoadbyFold(val_fold, feature_path, use_type = useType)
    v_samples = vData_shape[0]
    
    data_shape = (numFrame, tData_shape[2], tData_shape[3])
    
    print('\nInput shape: %s' % str(data_shape))
    
    ############### Define Model ###############
    initLR = 0.001
    num_middle_node = 512
    k_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
    model_name = 'CNN_model_(' + fold + ')_' + str(useType) + '_' + str(use_time_range)
    
    model = Sequential()
    model.add(Conv2D(32, (5,3), activation = 'selu', input_shape=data_shape,
                     kernel_initializer=k_init))
    model.add(Conv2D(32, (1,3), activation = 'selu', kernel_initializer=k_init))                 
    model.add(MaxPooling2D(pool_size = (1, 2))) 
    
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(num_middle_node, activation = 'selu', kernel_initializer=k_init))
    model.add(Dropout(0.5))
    model.add(Dense(num_middle_node, activation = 'selu', kernel_initializer=k_init))
    model.add(Dropout(0.5))
    model.add(Dense(88, activation = 'sigmoid'))
    
    
    optim = keras.optimizers.Adam(lr=initLR)
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=optim,
                  metrics=[st.Precision, st.Recall, st.Fscore])
    
    ############### Training Field ###############
    epoch = 20  #max epochs for training
    batchSize = 100
    v_batchSize = 500
    break_patience = 4  #early stop epochs. If val_loss doesn't decrease for given epochs, the training will stop.
    t_idx = np.random.permutation(t_samples)
    v_idx = np.random.permutation(v_samples)
    train_batches = int(np.ceil(t_samples/batchSize))
    val_batches = int(np.ceil(v_samples/v_batchSize))
    history = {}
    history['loss'] = []
    history['Fscore'] = []
    history['val_loss'] = []
    history['val_Fscore'] = []
    
    patience = 0
    best_f = 0
    best_v = 100000
    best_epoch = 0
    
    print('Using ' + str(useType) + ' for training')
    print('\nTrain on %d samples - Validate on %d samples' % (t_samples, v_samples))
    for e in range(epoch):
        cur_ep = 'Epoch %d/%d - ' % (e+1, epoch)
        loss = 0
        fscore = 0
        recall = 0
        precision = 0
        for i in range(train_batches):
            sel_t = t_idx[i*batchSize:(i+1)*batchSize]
            if i == train_batches-1:
                sel_t = t_idx[i*batchSize:]
            
            data, label = preCNN_processBatch(train_data, train_label, train_index, sel_t, numFrame)
            scalar = model.train_on_batch(data, label)
            
            loss += scalar[0]
            fscore += scalar[3]
            recall += scalar[2]
            precision += scalar[1]
            
            batch_info = cur_ep + '%d/%d - ' % (i+1, train_batches)
            batch_info += 'loss: %.4f - Precision: %.4f - Recall: %.4f - Fscore: %.4f' % (loss/(i+1), precision/(i+1), recall/(i+1), fscore/(i+1))
            if i == (train_batches-1):
                print(batch_info)
                history['loss'].append(loss/train_batches)
                history['Fscore'].append(fscore/train_batches)
            else:
                print(batch_info, end='\r')
            
        ### Validation ###
        v_loss = 0
        v_fscore = 0
        for i in range(val_batches):
            print('Validation progress: %d/%d' % (i+1, val_batches), end='\r')
            
            sel_v = v_idx[i*v_batchSize:(i+1)*v_batchSize]
            if i == val_batches-1:
                sel_v = v_idx[i*v_batchSize:]
            
            data, label = preCNN_processBatch(val_data, val_label, val_index, sel_v, numFrame)
            #data = preCNN(combineFrame_2(val_data, numFrame, sample_per_song=3000, sel_t), channel=3)[:,:,:,use_specific_channel]
            
            v_scalar = model.test_on_batch(data, label)
            
            v_loss += v_scalar[0]
            v_fscore += v_scalar[3]
        print('Validation - loss: %.4f - F-score: %.4f' % (v_loss/val_batches, v_fscore/val_batches))
        print('-------------------------------------------------------------------------------------------')
        
        history['val_loss'].append(v_loss/val_batches)
        history['val_Fscore'].append(v_fscore/val_batches)
        
        ### Early Stop Check (according to val_loss) ###
        if v_loss >= best_v:
            patience += 1
        else:
            patience = 0
            best_epoch = e
            best_v = v_loss
            best_f = v_fscore
            model.save('./Result/%s.hdf5' % model_name)
        if patience == break_patience:
            print('Early Stopped')
            break
        
    info = 'Best epoch: %d - Loss: %.4f - F-score: %.4f\n' % (best_epoch+1, (best_v/val_batches), (best_f/val_batches))
    print(info)
    log = open('log.txt', 'a')
    log.write(fold+str(useType)+'\n')
    log.write(info+'\n\n')
    log.close()
    
    
    #Save training and validation history
    out_history = h5py.File("History/CNN_history("+fold+str(useType)+").hd5", 'w')
    out_history.create_dataset('loss', data=history['loss'])
    out_history.create_dataset('Fscore', data=history['Fscore'])
    out_history.create_dataset('val_loss', data=history['val_loss'])
    out_history.create_dataset('val_Fscore', data=history['val_Fscore'])
    out_history.close()
    
    return info, model

def runCNN_Test(model, feature_path, useType=['Spec', 'Ceps', 'GCoS'], fold='test_config2'):
    numFrames = 5
    test_batchSize = 2000
    test_fold = fold+'.txt'
    
    Data, Label, Index, shape = LoadbyFold(test_fold, feature_path, use_type = useType)

    print('\nTesting on %d samples.' % shape[0])

    tp = 0
    fp = 0
    fn = 0
    idx = np.random.permutation(shape[0])
    batches = int(np.ceil(shape[0]/test_batchSize))

    for i in range(batches):
        sel_t = idx[i*test_batchSize:(i+1)*test_batchSize]
        if i == batches-1:
            sel_t = idx[i*test_batchSize:]
            
        data, label = preCNN_processBatch(Data,Label, Index, sel_t, numFrames)
        
        pred = model.predict(data)
        pred = np.where(pred>0.5, 1, 0)
        
        tp += sum(label[pred>0])
        fp += sum(pred[label<=0])
        fn += sum(label[pred<=0])
        
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        fscore = 2*precision*recall / (precision+recall)
        
        info = 'Progress: %d/%d - Precision: %.4f - Recall: %.4f - F-score: %.4f' % (i+1, batches, precision, recall, fscore)
        print(info, end='\r')
    
    return info


if __name__=='__main__':
    train_fold = 'config2_fold_'
    feature_path = 'I:/MAPS_DATASET/Full Extraction/'
    test_fold = 'test_config2'
    
    for i in range(4):
        fold = train_fold+str(i+1)
        use_specific_channel = [0, 1, 2]  #which feature type to use (0: spec, 1: ceps, 2: gcos)
        use_time_range = None
        #use_time_range = (0, 60)
        
        channel_type = ['Spec', 'Ceps', 'GCoS']
        useType = [channel_type[i] for i in use_specific_channel]

        start_t = time.time()
        train_info, model = runCNN_Train(feature_path, fold, useType, use_time_range)
        exec_time = time.time()-start_t
        
        test_info = runCNN_Test(model, feature_path, useType, test_fold)

        print('Training result: \n' + train_info)
        print('Training time: %.4f sec' % exec_time)
        print('Testing result: \n' + test_info + '\n\n')
        log = open('CNN_log.txt','a')
        log.write(fold+str(useType)+'_'+str(use_time_range))
        log.write('\nTraining result:\n' + train_info + 'Training time: %.4f' % exec_time)
        log.write('\nTesting result:\n' + test_info + '\n\n')
        log.close()




  
