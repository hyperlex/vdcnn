#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
import os
import time
import datetime
import tables
from sklearn.metrics import f1_score,confusion_matrix



# =====================  Preparation des données    =============================

# Load data
print("Loading data...")

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
sequence_max_length = 1024 # Twitter has only 140 characters. We pad 4 blanks characters more to the right of tweets to be conformed with the architecture of A. Conneau et al (2016)

from tensorflow.core.protobuf import saver_pb2

checkpoint_file = tf.train.latest_checkpoint("./")

graph = tf.Graph()
# Input data.
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
	# Get the placeholders from the graph by name
        
	input_x = graph.get_operation_by_name("input_x").outputs[0]
	input_y = graph.get_operation_by_name("input_y").outputs[0]

	is_training = graph.get_operation_by_name(
            "phase").outputs[0]

        ### To update the computation of moving_mean & moving_var, we must put it on the parent graph of minimizing loss

        accuracy = graph.get_operation_by_name(
            "accuracy/accuracy").outputs[0]

        predictions = graph.get_operation_by_name(
            "fc-3/predictions").outputs[0]
 	hdf5_path = "my_extendable_compressed_data_test.hdf5"

	batch_size = 1000
        extendable_hdf5_file = tables.open_file(hdf5_path, mode='r')

        y_true_ = []
        predictions_= []

        for ptr in range(0, 70000, batch_size):

            feed_dict = {cnn.input_x: extendable_hdf5_file.root.data[ptr:ptr + batch_size], cnn.input_y: extendable_hdf5_file.root.clusters[ptr:ptr + batch_size] , cnn.is_training: False  }    

            y_true = tf.argmax(extendable_hdf5_file.root.clusters[ptr:ptr + batch_size] , 1)
            y_true_bis,predictions_bis ,accuracy = sess.run([y_true,predictions,cnn.accuracy], feed_dict= feed_dict)    
            y_true_.extend(y_true_bis)
            predictions_.extend(predictions_bis)


        confusion_matrix_ = confusion_matrix(y_true_,predictions_)
        print(confusion_matrix_)
        print ("f1_score", f1_score(y_true_, predictions_ ,average ='weighted'))
        print ("f1_score", f1_score(y_true_, predictions_ ,average =None))
        extendable_hdf5_file.close()
