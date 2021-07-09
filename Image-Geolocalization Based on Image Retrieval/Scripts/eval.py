import tensorflow as tf
import numpy as np
import os

import siamese
from input_data_Ours import InputData

# Hyper parameters
isHPM = 'HPM'
train_data_type = 'CVUSA'
model_type = 'ResNet50'
load_epoch = 49
batch_size = 16
loss_weight = 10.0

# TestData Parameters
SatLevel = "19"
GrdType = "Aligned"
test_data_type = SatLevel + "_" + GrdType

def validate(dist_array, top_k):
    accuracy = 0.0
    data_amount = 0.0
    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, i]
        prediction = np.sum(dist_array[:, i] < gt_dist)
        if prediction < top_k:
            accuracy += 1.0
        data_amount += 1.0
    accuracy /= data_amount
    return accuracy

def eval():
    input_data = InputData()

    # define placeholders
    with tf.name_scope('inputs'):
        sat_x = tf.placeholder(tf.float32, [None, 224, 224, 3], name='sat_img')
        grd_x = tf.placeholder(tf.float32, [None, 224, 224, 3], name='grd_img')

    learning_rate = tf.placeholder(tf.float32)

    # build the model
    if(model_type == 'VGG16'):
        sat_global, grd_global = siamese.siamese_VGG16(sat_x, grd_x)
    elif(model_type == 'ResNet50'):
        sat_global, grd_global = siamese.siamese_ResNet50(sat_x, grd_x)
    else:
        print("Please check the model type!")

    # define descriptors
    out_channel = sat_global.get_shape().as_list()[-1]
    sat_global_descriptor = np.zeros([input_data.get_test_dataset_size(), out_channel])
    grd_global_descriptor = np.zeros([input_data.get_test_dataset_size(), out_channel])

    # set saver
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    # training configurations
    print('set config...')
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    # eval
    print('open sess...')
    with tf.Session(config=config) as sess:
        print('initialization...')
        sess.run(tf.global_variables_initializer())

        # load model
        print('load model...')
        load_model_path = '../Model/' + train_data_type + '/' + model_type + '/' + isHPM + '/' + str(load_epoch) + '/model.ckpt'
        saver.restore(sess, load_model_path)
        print('     Model loaded from from: %s' % load_model_path)
        print("load model...FINISHED!")

        # validation
        print('validation...')
        print('     compute global descriptor')
        input_data.reset_scan()
        val_i = 0
        while True:
            print('     progress %d' % val_i)
            batch_sat, batch_grd = input_data.next_batch_scan(batch_size)
            if batch_sat is None:
                break

            feed_dict = {sat_x: batch_sat, grd_x: batch_grd}
            sat_global_val, grd_global_val = sess.run([sat_global, grd_global], feed_dict=feed_dict)

            sat_global_descriptor[val_i: val_i + sat_global_val.shape[0], :] = sat_global_val
            grd_global_descriptor[val_i: val_i + grd_global_val.shape[0], :] = grd_global_val
            val_i += sat_global_val.shape[0]

        print('compute accuracy...')
        dist_array = 2 - 2 * np.matmul(sat_global_descriptor, np.transpose(grd_global_descriptor))
        # top1_percent = int(dist_array.shape[0] * 0.01) + 1 # for CVUSA or CVACT
        top1_percent = input_data.get_test_dataset_size() # for small dataset like CVBIT/CVLIT
        val_accuracy = np.zeros((1, top1_percent))
        for i in range(top1_percent):
            val_accuracy[0, i] = validate(dist_array, i)
        print("train_data_type: %s model_type: %s model_epoch: %s test_data_type: %s isHPM: %s" % (train_data_type, model_type, load_epoch, test_data_type, isHPM))
        print('     top1', ':', val_accuracy[0, 1])
        print('     top5', ':', val_accuracy[0, 5])
        print('     top10', ':', val_accuracy[0, 10])
        print('     top1%', ':', val_accuracy[0, -1])

        print('save descriptor...')
        np_save_dir = '../Visualize/' + model_type + '/' + test_data_type + "/"  + isHPM +  '/'
        if not os.path.exists(np_save_dir):
            os.makedirs(np_save_dir)
        np.save(np_save_dir + 'dist_array.npy', dist_array)
        np.save(np_save_dir + 'recall_accuracy.npy', val_accuracy)
        print('dist_array and val_accuracy saved!')

if __name__ == '__main__':
    eval()















