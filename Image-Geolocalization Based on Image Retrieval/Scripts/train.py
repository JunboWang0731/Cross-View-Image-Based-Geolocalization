import tensorflow as tf
import numpy as np
import os

import siamese

# Hyper parameters
isHPM = 'HPM'
data_type = 'CVACT'
model_type = 'VGG16'
start_epoch = 0
number_of_epoch = 51

batch_size = 16
learning_rate_val = 1e-5
loss_weight = 10.0

if data_type == 'CVUSA':
    from input_data_CVUSA import InputData
elif data_type == 'CVACT':
    from input_data_CVACT import InputData
else:
    print("Invalid input_data!")

def validate(grd_descriptor, sat_descriptor):
    accuracy = 0.0
    data_amount = 0.0
    dist_array = 2 - 2 * np.matmul(sat_descriptor, np.transpose(grd_descriptor))
    top1_percent = int(dist_array.shape[0] * 0.01) + 1
    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, i]
        prediction = np.sum(dist_array[:, i] < gt_dist)
        if prediction < top1_percent:
            accuracy += 1.0
        data_amount += 1.0
    accuracy /= data_amount

    return accuracy

def triplet_loss(sat_global, grd_global, batch_hard_count=0):
    with tf.name_scope('weighted_soft_margin_triplet_loss'):
        dist_array = 2 - 2 * tf.matmul(sat_global, grd_global, transpose_b=True)
        pos_dist = tf.diag_part(dist_array)
        if batch_hard_count == 0:
            pair_n = batch_size * (batch_size - 1.0)

            # ground to satellite
            triplet_dist_g2s = pos_dist - dist_array
            loss_g2s = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_g2s * loss_weight))) / pair_n

            # satellite to ground
            triplet_dist_s2g = tf.expand_dims(pos_dist, 1) - dist_array
            loss_s2g = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_s2g * loss_weight))) / pair_n

            loss = (loss_g2s + loss_s2g) / 2.0

        else:
            # ground to satellite
            triplet_dist_g2s = pos_dist - dist_array
            triplet_dist_g2s = tf.log(1 + tf.exp(triplet_dist_g2s * loss_weight))
            top_k_g2s, _ = tf.nn.top_k(tf.transpose(triplet_dist_g2s), batch_hard_count)
            loss_g2s = tf.reduce_mean(top_k_g2s)

            # satellite to ground
            triplet_dist_s2g = tf.expand_dims(pos_dist, 1) - dist_array
            triplet_dist_s2g = tf.log(1 + tf.exp(triplet_dist_s2g * loss_weight))
            top_k_s2g, _ = tf.nn.top_k(triplet_dist_s2g, batch_hard_count)
            loss_s2g = tf.reduce_mean(top_k_s2g)

            loss = (loss_g2s + loss_s2g) / 2.0

    return loss

# if no pretrained model, use start_epoch = 0
def train():
    print("data_type: %s" % data_type)
    input_data = InputData(HPM=isHPM)

    # define placeholders
    with tf.name_scope('inputs'):
        sat_x = tf.placeholder(tf.float32, [None, 224, 224, 3], name='sat_img')
        grd_x = tf.placeholder(tf.float32, [None, 224, 224, 3], name='grd_img')

    learning_rate = tf.placeholder(tf.float32)

    # build the model
    if model_type == 'VGG16':
        sat_global, grd_global = siamese.siamese_VGG16(sat_x, grd_x)
    elif model_type == 'ResNet50':
        sat_global, grd_global = siamese.siamese_ResNet50(sat_x, grd_x)
    else:
        print("Error! invalid model type!")

    # define descriptors
    out_channel = sat_global.get_shape().as_list()[-1]
    sat_global_descriptor = np.zeros([input_data.get_test_dataset_size(), out_channel])
    grd_global_descriptor = np.zeros([input_data.get_test_dataset_size(), out_channel])

    # define loss
    loss = triplet_loss(sat_global, grd_global, 0)
    tf.summary.scalar('loss', loss)

    # set optimizer
    global_step = tf.Variable(0, trainable=False)
    with tf.device('/gpu:0'):
        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(learning_rate, 0.9, 0.999).minimize(loss, global_step=global_step)

    # set saver
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

    # training configurations
    print('run model...')
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95

    # train

    with tf.Session(config=config) as sess:
        # record training data
        merge_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('../logs/' + data_type + '/' + model_type + '/' + isHPM + '/', sess.graph)

        # initialize
        sess.run(tf.global_variables_initializer())

        if start_epoch != 0:
            print('load model...')
            load_model_path = '../Model/' + data_type + '/' + model_type + '/' + isHPM + '/' + str(start_epoch - 1) + '/model.ckpt'
            saver.restore(sess, load_model_path)
            print("   Model loaded from: %s" % load_model_path)
            print('load model...FINISHED')
        else:
            print('No model loaded, Start initial traing')

        for epoch in range(start_epoch, start_epoch + number_of_epoch):
            iter = 0
            while True:
                # train
                batch_sat, batch_grd = input_data.next_pair_batch(batch_size)
                if batch_sat is None:
                    break

                global_step_val = tf.train.global_step(sess, global_step)
                feed_dict = {sat_x: batch_sat, grd_x: batch_grd, learning_rate: learning_rate_val}

                if iter % 20 == 0:
                    _, loss_val = sess.run([train_step, loss], feed_dict=feed_dict)
                    print('global %d, epoch %d, iter %d: loss: %.4f' % (global_step_val, epoch, iter, loss_val))
                    result = sess.run(merge_summary, feed_dict=feed_dict)
                    writer.add_summary(result, global_step_val)
                else:
                    sess.run(train_step, feed_dict=feed_dict)

                iter += 1

            # validation
            print('validation...')
            print('compute global descriptor')
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

            print('     compute accuracy')
            val_accuracy = validate(grd_global_descriptor, sat_global_descriptor)
            print('     %d: accuracy = %.1f%%' % (epoch, val_accuracy*100.0))

            # save accuracy in txt file and save model
            file_dir = '../Model/' + data_type + '/' + model_type + '/' + isHPM + '/'
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            with open(file_dir + 'accuracy.txt', 'a') as file:
                file.write(str(epoch) + ':' + str(val_accuracy) + '\n')

            model_dir = '../Model/' + data_type + '/' + model_type + '/' + isHPM + '/' + str(epoch) + '/'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            save_path = saver.save(sess, model_dir + 'model.ckpt')
            print("Model saved in file: %s" % save_path)

if __name__ == '__main__':
    train()















