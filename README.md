# vdcnn

Based on the paper" Very Deep Convolutional Networks for Natural Language Processing
           "(https://arxiv.org/abs/1606.01781) in tensorflow. 

Model parameters are initialized using the Xavier Glorot and Yoshua Bengio (2010) initializer:
           [Understanding the difficulty of training deep feedforward neural
           networks. International conference on artificial intelligence and
           statistics.](
           http://www.jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)

Model trained and tested on a Tesla K80 gpu: 
                    - 32min per epoch 
                    - Test accuracy of 0.95 
