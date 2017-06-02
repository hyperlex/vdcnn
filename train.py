#!/usr/bin/env python
# -*- coding: utf-8 -*-

# based on ideas in https://github.com/lethienhoa/Very-Deep-Convolutional-Networks-for-Natural-Language-Processing/blob/master/train.py

import tensorflow as tf 
from vdcnn import VDCNN
import numpy as np
import os
import time
import datetime
import cPickle as pkl
import tables


# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 400, "Batch Size (default: 128)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 5000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# =====================  Preparation des données    =============================

# Load data
print("Loading data...")

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
sequence_max_length = 1024

# shuffeling data for training 

# Training
# ==================================================

# ----------------- Phase de construction du graphe -------------------------------



# Input data.
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = VDCNN()
	# Ensures that we execute the update_ops before performing the train
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
		global_step = tf.Variable(0, name="global_step", trainable=False)
		optimizer = tf.train.AdamOptimizer(1e-3)
		grads_and_vars = optimizer.compute_gradients(cnn.loss)
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


        # Initialize all variables

	print("START %s" % datetime.datetime.now())
        sess.run(tf.initialize_all_variables())
	
	saver = tf.train.Saver()
	print('Initialized')
	
	batch_size = FLAGS.batch_size
	epochs = FLAGS.num_epochs
	hdf5_path = "my_extendable_compressed_data_train.hdf5"
	for e in range(epochs):

                extendable_hdf5_file = tables.open_file(hdf5_path, mode='r')
	        for ptr in range(0, 500000, batch_size):
			#print(ptr)
			feed_dict = {
   			   cnn.input_x: extendable_hdf5_file.root.data[ptr: ptr+batch_size],
             		   cnn.input_y:extendable_hdf5_file.root.clusters[ptr: ptr+batch_size] ,
              		   cnn.is_training: True	}	# Update moving_mean, moving_var }

        		sess.run(train_op,feed_dict)
			time_str = datetime.datetime.now().isoformat()
 	        if e % 1 == 0:
                        step ,loss, accuracy =  sess.run([global_step, cnn.loss, cnn.accuracy],feed_dict)
            		save_path = saver.save(sess, "model_vdcnn_full_dataset.ckpt")
            		print("model saved in file: %s" % save_path)
			print("{}: epoch {}, loss {}, acc {}".format(time_str,e, loss, accuracy))

                print("epoch %d:" % e)
	        extendable_hdf5_file.close()

	print("END %s" % str(datetime.datetime.now()))
