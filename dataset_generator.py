#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import csv
import tables



alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
sequence_max_length = 1024
num_classes = 14


def pad_sentence(char_seq, padding_char=" "):
    num_padding = sequence_max_length - len(char_seq)
    new_char_seq = char_seq + [padding_char] * num_padding
    return new_char_seq



def string_to_int8_conversion(char_seq, alphabet):
    x = np.array([alphabet.find(char) for char in char_seq], dtype=np.int8)
    return x



def get_one_hot_embedding(char_sequence_indice):
        y_one_hot = np.zeros(shape=[len(char_sequence_indice),len(alphabet)])
        for num_index,char_indice in enumerate(char_sequence_indice):
                y_one_hot[num_index][char_indice] = 1

        return  y_one_hot

def get_one_hot_label(class_index):
    y_one_hot = np.zeros(shape=[num_classes])
    y_one_hot[class_index -1] = 1

    return y_one_hot


train_inp = np.empty([1,sequence_max_length, len(alphabet)],dtype = 'float64')
train_out = np.empty([1,num_classes], dtype = 'float64')

train_inp_ = []
train_out_ = []


hdf5_path = "data/my_extendable_compressed_data_train.hdf5"

hdf5_file = tables.open_file(hdf5_path, mode='w')
filters = tables.Filters(complevel=5, complib='blosc')
data_storage = hdf5_file.create_earray(hdf5_file.root, 'data',
                                      tables.Atom.from_dtype(train_inp.dtype), #.dtype),
                                      shape=(0,sequence_max_length ,len(alphabet)),
                                      filters=filters,
                                      expectedrows=len(train_inp))
clusters_storage = hdf5_file.create_earray(hdf5_file.root, 'clusters',
                                          tables.Atom.from_dtype(train_out.dtype),
                                          shape=(0,num_classes),
                                          filters=filters,
                                          expectedrows=len(train_inp))

hdf5_file.close()


# Load data from csv file
print("Loading training data...")
cmp =0

with open('dbpedia_csv/random.csv') as f:
    reader = csv.DictReader(f,fieldnames=['class'],restkey='fields')
    for row in reader:
        cmp += 1
        padded_input = pad_sentence(list((row['fields'][1]).lower())[:sequence_max_length])
        char_sequence_index = string_to_int8_conversion(padded_input, alphabet)
        embedding = get_one_hot_embedding(char_sequence_index)
        label = get_one_hot_label(int(row['class']))
        
        if (cmp == 2000) : #or (cmp == len(reader)-1 ):
            extendable_hdf5_file = tables.open_file(hdf5_path, mode='a')
            extendable_hdf5_data = extendable_hdf5_file.root.data
            extendable_hdf5_clusters = extendable_hdf5_file.root.clusters
            print("Length of current data: %i" % len(extendable_hdf5_data))
            print("Length of current cluster labels: %i" % len(extendable_hdf5_clusters))

	    #adaptation
            train_inp_.append(embedding)
            train_out_.append(label)

	    train_inp = np.asarray(train_inp_)
	    train_out = np.asarray(train_out_)

	    print(train_inp.shape)
            for n, (d, c) in enumerate(zip(train_inp[:], train_out[:])):
                    extendable_hdf5_data.append(train_inp[n][None])
                    extendable_hdf5_clusters.append(train_out[n][None])
            extendable_hdf5_file.close()
 
            train_inp = np.empty([1,sequence_max_length, len(alphabet)],dtype = 'float64')
            train_out = np.empty([1,num_classes], dtype = 'float64')
	    train_inp_ = []
	    train_out_ = []
            cmp =0
        else:
	    train_inp_.append(embedding)
	    train_out_.append(label)




test_inp_ = []
test_out_ = []


# Load data from csv file
print("Loading test data...")

test_inp = np.empty([1,sequence_max_length, len(alphabet)],dtype = 'float64')
test_out = np.empty([1,num_classes], dtype = 'float64')

hdf5_path = "data/my_extendable_compressed_data_test.hdf5"

hdf5_file = tables.open_file(hdf5_path, mode='w')
filters = tables.Filters(complevel=5, complib='blosc')
data_storage = hdf5_file.create_earray(hdf5_file.root, 'data',
                                      tables.Atom.from_dtype(test_inp.dtype), #.dtype),
                                      shape=(0,sequence_max_length ,len(alphabet)),
                                      filters=filters,
                                      expectedrows=len(test_inp))
clusters_storage = hdf5_file.create_earray(hdf5_file.root, 'clusters',
                                          tables.Atom.from_dtype(test_out.dtype),
                                          shape=(0,num_classes),
                                          filters=filters,
                                          expectedrows=len(test_inp))


hdf5_file.close()


cmp =0

with open('dbpedia_csv/test.csv') as f:
    reader = csv.DictReader(f,fieldnames=['class'],restkey='fields')
    for row in reader:
        cmp += 1
        padded_input = pad_sentence(list((row['fields'][1]).lower())[:sequence_max_length])
        char_sequence_index = string_to_int8_conversion(padded_input, alphabet)
        embedding = get_one_hot_embedding(char_sequence_index)
        label = get_one_hot_label(int(row['class']))

        if (cmp == 1000) : #or (cmp == len(reader)-1 ):
            extendable_hdf5_file = tables.open_file(hdf5_path, mode='a')
            extendable_hdf5_data = extendable_hdf5_file.root.data
            extendable_hdf5_clusters = extendable_hdf5_file.root.clusters
            print("Length of current data: %i" % len(extendable_hdf5_data))
            print("Length of current cluster labels: %i" % len(extendable_hdf5_clusters))

            train_inp_.append(embedding)
            train_out_.append(label)

            test_inp = np.asarray(test_inp_)
            test_out = np.asarray(test_out_)


            for n, (d, c) in enumerate(zip(test_inp[:], test_out[:])):
                    extendable_hdf5_data.append(test_inp[n][None])
                    extendable_hdf5_clusters.append(test_out[n][None])
            extendable_hdf5_file.close()
            
	    test_inp = np.empty([1,sequence_max_length, len(alphabet)],dtype = 'float64')
            test_out = np.empty([1,num_classes], dtype = 'float64')
            test_inp_ = []
            test_out_ = []

            cmp =0
        else:
            test_inp_.append(embedding)
            test_out_.append(label)

