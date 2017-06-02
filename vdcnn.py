#!/usr/bin/env python
# -*- coding: utf-8 -*-

# based on ideas from https://github.com/lethienhoa/Very-Deep-Convolutional-Networks-for-Natural-Language-Processing/


import tensorflow as tf

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
sequence_max_length = 1024 
top_k_max_pooling_size = 8

class VDCNN():
 
    
    # ----- Based on the Very Deep Convolutional Networks for Natural Language Processing paper. -----
    

    def __init__(self, num_classes=14, cnn_filter_size=3, pooling_filter_size=2, num_filters_per_size=(64,128,256,512), 
			num_rep_block=(1,1,1,1), num_quantized_chars=len(alphabet), sequence_max_length=sequence_max_length, l2_reg_lambda=0.005):
  
        self.input_x = tf.placeholder(tf.float32, [None, sequence_max_length, num_quantized_chars ], name="input_x")		
	input_x_  = tf.transpose(self.input_x,[0,2,1])
	input_x_ = tf.expand_dims(input_x_, -1)
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.is_training = tf.placeholder(tf.bool, name="phase")


	

        # l2-regularization loss 
        l2_loss = tf.constant(0.0)

        # ================ First Conv Layer ================
        with tf.name_scope("first-conv-layer"):
            filter_shape = [num_quantized_chars, cnn_filter_size, 1, num_filters_per_size[0]]	
	    initializer = tf.contrib.layers.xavier_initializer()
            W = tf.Variable(initializer(shape=filter_shape), name="W")
	    b = tf.Variable(tf.constant(0.1, shape=[num_filters_per_size[0]]), name="b")
            conv = tf.nn.conv2d(input_x_, W, strides=[1, num_quantized_chars, 1, 1], padding="SAME", name="first-conv")	
            h = tf.nn.bias_add(conv, b)

        # ================ Conv Block 64, 128, 256, 512 ================
	for i in range(0,4):
		with tf.name_scope("conv-block-%s"%i):
		    for j in range(0,num_rep_block[i]):
			with tf.name_scope("sub1-%s"%j):
                            filter_shape = [1,cnn_filter_size,int(h.get_shape()[3]), num_filters_per_size[i]]
                            initializer = tf.contrib.layers.xavier_initializer_conv2d()
    			    W1 = tf.Variable(initializer(shape=filter_shape), name="W1")
			    conv1 = tf.nn.conv2d(h, W1, strides=[1, 1, 1, 1], padding="SAME", name="conv1")	
			    batch_norm1 = tf.contrib.layers.batch_norm(conv1,                                          
									  center=True, scale=True, decay=0.9, 
									  is_training=self.is_training) 
                            batch_norm1 = tf.identity(batch_norm1, name="bn1")

			    h1 = tf.nn.relu(batch_norm1, name="relu1")

			with tf.name_scope("sub2"):
	                    filter_shape2 = [1,cnn_filter_size,num_filters_per_size[i], num_filters_per_size[i]]
                            initializer = tf.contrib.layers.xavier_initializer_conv2d()
                            W2 = tf.Variable(initializer(shape=filter_shape2), name="W2")
			    conv2 = tf.nn.conv2d(h1, W2, strides=[1, 1, 1, 1], padding="SAME", name="conv2")	
			    batch_norm2 = tf.contrib.layers.batch_norm(conv2,                                          
									  center=True, scale=True, decay=0.9, 
									  is_training=self.is_training)  #, name='bn2')
			    batch_norm2 = tf.identity(batch_norm2, name="bn2")
			    h = tf.nn.relu(batch_norm2, name="relu2")

		if (i<>3):	
		    with tf.name_scope("max-pooling"):
			h = tf.nn.max_pool(h, ksize=[1, 1, pooling_filter_size,1 ],
		        		      strides=[1, 1, 2, 1], padding='SAME', name="pool")

	# ================ Top k-max pooling ================
	h1 = tf.transpose(h,[0,3,1,2])
	top_k_max_pooling = tf.nn.top_k(h1, k=top_k_max_pooling_size)
	top_k_max_pooling = top_k_max_pooling 
	
	h_pool_flat = tf.reshape(top_k_max_pooling[0],(-1,512*8))

        # ================ Layer FC 1 ================

        # Fully connected layer 1
        with tf.name_scope("fc-1"):
            initializer = tf.contrib.layers.xavier_initializer()
            W = tf.Variable(initializer(shape=[ 4096,2048]), name="W")
	    b = tf.Variable(tf.constant(0.1, shape=[2048]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            fc_1_output = tf.nn.relu(tf.nn.xw_plus_b(h_pool_flat, W, b), name="fc-1-out")


        # ================ Layer FC 2 ================

        # Fully connected layer 2
        with tf.name_scope("fc-2"):
            initializer = tf.contrib.layers.xavier_initializer()
            W = tf.Variable(initializer(shape=[2048, 2048]), name="W")
	    b = tf.Variable(tf.constant(0.1, shape=[2048]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            fc_2_output = tf.nn.relu(tf.nn.xw_plus_b(fc_1_output, W, b), name="fc-2-out")


        # ================ Layer FC 3 ================
 
        # Fully connected layer 3
        with tf.name_scope("fc-3"):

            W = tf.Variable(tf.truncated_normal([2048, num_classes], stddev=0.05), name="W")            
	    b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            scores = tf.nn.xw_plus_b(fc_2_output, W, b, name="output")
            predictions = tf.argmax(scores, 1, name="predictions")
        
	# ================ Loss and Accuracy ================
        
	with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels= self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
