from keras import backend as K

def Precision(label,pred):
    true_positives = K.sum(K.round(K.clip(label * pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(pred, 0, 1)))
    precision = true_positives / predicted_positives
    return precision
    
def Recall(label,pred):
    true_positives = K.sum(K.round(K.clip(label * pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(label, 0, 1)))
    recall = true_positives / possible_positives
    return recall

def Fscore(label,pred):
    true_positives = K.sum(K.round(K.clip(label * pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(pred, 0, 1)))
    precision = true_positives / predicted_positives
    
    possible_positives = K.sum(K.round(K.clip(label, 0, 1)))
    recall = true_positives / possible_positives
    
    fscore = 2*precision*recall / (precision+recall)
    
    return fscore

def cal_stat(pred, label):
    tp = sum(label[pred>0])
    fp = sum(pred[label<=0])
    fn = sum(label[pred<=0])
    
    return tp, fp, fn

def get_fscore(pred, label):
    tp, fp, fn = cal_stat(pred, label)
    
    fscore = 2*tp / (2*tp+fp+fn)
    
    return fscore







