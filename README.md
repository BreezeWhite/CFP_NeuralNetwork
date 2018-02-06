## CFP Neural Network 

This is the code used in the following paper: <br />
**Yu-Te Wu, Berlin Chen, and Li Su, "Automatic music transcription leveraging generalized cepstral features and deep learning," IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), April 2018**

There are two parts of the code: one is for feature extraction; and the another one is for NN construction, training, and testing. To run the code, you need to install keras with tensorflow backend. You can either use GPU for training or not. Just modify the code in **_FullTrainTest.py,  line 12*, set the os environment variable equals to "" for not to use GPU.

Before run the training code, make sure that you have already done the feature extraction. If not, run the *CFP_Extraction.m* for generating the necessary files. Or you can also run the python version: *GenFeature.py*. But you have to write additional code for processing through all the dataset automatically. Also, you have to modify some path variables inside the **_FullTrainTest.py* file. Set it to your path where your files are.

The structure of CNN model is visulized as below:
![](/home/whitebreeze/Document/CNN_Model.jpg) 
And the structure of DNN is just the last four layers of CNN.

The full testing result reported in the paper:<br />
[Test Result](https://drive.google.com/open?id=1semWG4RHFSFDoH21fzQzY-eY9e-wRT3WjG9MDkXmZiE) 

Enjoy~