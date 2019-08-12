from tensorflow.contrib.layers import convolution2d, batch_norm, max_pool2d, fully_connected
from time import time
import tensorflow as tf
import numpy as np
import pdb
import load_NYU_Data as loaddata
import PIL.Image as im
import matplotlib.pyplot as plt
import random
import math
import os
import cv2

# 462032 afbeeldingen waarvan 460000 afbeeldingen zijn gerandomiseerd in testh5py.hdf5
# de laatste 2000 afbeeldingen kunnen uit de gewone NUY_lzif.hdf5 file gehaald worden als testset.

tf.flags.DEFINE_string("dir", "/users/start2012/r0298867/Thesis/implementation1/tmp3/", "main directory.")
tf.flags.DEFINE_float("learning_rate", 0.01, "learning rate.")
tf.flags.DEFINE_integer("num_epochs", 1, "number of epochs.")
tf.flags.DEFINE_integer("num_batches", 20000, "number of batches.")
tf.flags.DEFINE_boolean("fine_scale", False, "Train coarse or fine-scale network.")
tf.flags.DEFINE_boolean("train", True, "Train network or evaluate when false")

FLAGS = tf.flags.FLAGS

'''
# The only thing that needs changing when switching between training coarse or fine-scale is  session.run([optimizer_all] -> session.run([optimizer] or session.run([optimizer_fs]
'''

def load_data():
    [dataGenerator, config_file] =  loaddata.main()
    return dataGenerator, config_file

def create_depth_image(depthmap, groundtruth):
    '''
    Depthmap is log scale, this function will recreate depth image by calculating exponential and normalizing.
    First normalize between 0 and 1, then rescale to absolute depth values using the groundtruth.
    Both depthmap en groundtruth are size [batchsize, 55*74]
    '''
    depthmap_exp = tf.exp(depthmap)
    depthmap_norm = tf.divide(depthmap_exp - tf.reduce_min(depthmap_exp), tf.reduce_max(depthmap_exp) - tf.reduce_min(depthmap_exp))
    depthmap_abs = depthmap_norm * (tf.reduce_max(groundtruth) - tf.reduce_min(groundtruth)) + tf.reduce_min(groundtruth)

    return depthmap_abs

def scale_invariant_loss(depthmap, groundtruth, pixels):
    lambd = 0.5
    #d = depthmap - tf.log(groundtruth+1e-10)
    d = (depthmap)-tf.log(groundtruth+1e-10)
    n = pixels
    var1 = tf.reduce_sum(tf.pow(d,2),1,keep_dims=True)
    var2 = tf.pow(tf.reduce_sum(d,1,keep_dims=True),2)
    loss = var1/n - lambd*var2/(n**2)
    loss = tf.reduce_sum(loss)

    return loss

def treshold(depthmap, groundtruth, pixels):
    '''
    Shape depthmap: depthmap.shape = (20,4070)
    Shape groundtruth: groundtruth.shape = (20,4070)
    pixels = Hout*Wout*batchsize  (55*74*20) 
    '''
    treshold = 1.25

    div_op1 = tf.divide(depthmap, groundtruth, name='Div1')
    div_op2 = tf.divide(groundtruth, depthmap, name='Div2')

    max_op = tf.maximum(div_op1, div_op2, name='Max')
    percentage = tf.divide( tf.reduce_sum(tf.cast(tf.greater(treshold, max_op), tf.int32)), pixels, name='Div_percentage')
    # percentage =  tf.reduce_sum(tf.cast(tf.greater(treshold, max_op), tf.int32))

    return percentage

def RMSE_linear(depthmap, groundtruth, pixels):
    '''
    Shape depthmap: depthmap.shape = (20,4070)
    Shape groundtruth: groundtruth.shape = (20,4070)
    pixels = Hout*Wout*batchsize  (55*74*20) 
    '''

    n = pixels
    RMSE = tf.reduce_sum( tf.pow(depthmap - groundtruth,2) ) / (n)
    RMSE = tf.sqrt(RMSE)
    return RMSE

def RMSE_log_inv(depthmap, groundtruth, pixels):
    '''
    Shape depthmap: depthmap.shape = (20,4070)
    Shape groundtruth: groundtruth.shape = (20,4070)
    pixels = Hout*Wout*batchsize  (55*74*20) 
    '''

    n = pixels
    alpha = tf.reduce_sum(tf.log(groundtruth + 1e-10) - tf.log(depthmap+1e-10)) / n #reduce_mean
    RMSE = tf.reduce_sum(tf.pow(tf.log(depthmap+1e-10) - tf.log(groundtruth + 1e-10) + alpha,2)) / (n)
    RMSE = tf.sqrt(RMSE)
    return RMSE
    # pdb.set_trace()

    #gt,depth = session.run([ground_truth, depth_map_fs],feed_dict={tf_train_dataset: val_rgb, tf_train_labels:val_depth,phase:0,fs_phase:0})

def evaluation(depthprediction, groundtruth):
    rmse = tf.metrics.mean_squared_error(depthprediction, groundtruth)
    
    return rmse


def train(dataGenerator, config_file):
    learning_rate = float(FLAGS.learning_rate)
    batch_norm_decay = 0.99
    num_epochs = int(FLAGS.num_epochs)
    # Input sizes
    image_size_x = config_file['H']
    image_size_y = config_file['W']
    # Output sizes
    Hout = config_file['HOut']
    Wout = config_file['WOut']
    # Batch size
    batch_size = config_file['batchSize']
    # 20% van batches voor validation
    # total_batches = dataGenerator.batchAm
    # number_of_batches = int(math.floor(total_batches * 0.8))
    # number_of_val_batches = int(math.floor(total_batches * 0.2))
    # number_of_batches = int(math.floor(total_batches * 0.005))
    # number_of_val_batches = int(math.floor(total_batches * 0.001))
    number_of_batches = int(FLAGS.num_batches)
    number_of_val_batches= 2000
    fsbool = FLAGS.fine_scale
    trainbool = FLAGS.train

    print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
    print 'number of training batches %d' % number_of_batches
    print 'number of validation batches %d' % number_of_val_batches
    print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'

    # Random list
    # list = [i for i in range(number_of_batches)]
    # random_list = random.sample(list, len(list))
    # Number of input channels (RGB)
    num_channels = 3
    # Conv filter size / kernel size
    patch_size1 = 11
    patch_size2 = 5
    patch_size3 = 3
    fine1_patch_size = 9
    fine3_patch_size = 5
    fine4_patch_size = 5
    # Number of filters per layer
    depth1 = 96
    depth2 = 256
    depth3 = 384
    fine1depth = 63
    fine3depth = 64
    fine4depth = 1

    # TODO Output size of fully connected layer?
    num_hidden = 4096
    num_hidden2 = 4070

    # logdir = '/users/start2012/r0298867/Thesis/implementation1/tmp3/graph'
    # savedir = '/users/start2012/r0298867/Thesis/implementation1/tmp3/saver/test'
    # checkpoint = '/users/start2012/r0298867/Thesis/implementation1/tmp3/saver/'
    logdir = '%sgraph' % FLAGS.dir
    savedir = '%ssaver/test' % FLAGS.dir
    checkpoint = '%ssaver/' % FLAGS.dir

    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)

    graph = tf.Graph()

    with graph.as_default():


        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size_x, image_size_y, num_channels), name='dataset')
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, 55, 74), name='labels')
        phase = tf.placeholder(tf.bool, name='phase')
        global_step = tf.Variable(0, name='global_step', trainable=False)

        def model(data, phase):
            with tf.variable_scope('Coarse'):
                with tf.variable_scope('layer1'):
                    conv1 = tf.contrib.layers.conv2d(data, num_outputs=depth1, kernel_size=patch_size1, stride=4, padding='VALID', trainable=True) # , scope='conv1'
                    batch_norm_1 = tf.contrib.layers.batch_norm(conv1, decay=batch_norm_decay, is_training=phase, updates_collections=tf.GraphKeys.UPDATE_OPS) # scale=True ,scope='bn1'
                    pool1 = tf.contrib.layers.max_pool2d(batch_norm_1, kernel_size=2, stride=2, padding='VALID')
                    relu1 = tf.nn.relu(pool1, 'relu1')

                    shape1 = batch_norm_1.get_shape().as_list()
                    print 'shape1 :',shape1

                with tf.variable_scope('layer2'):
                    conv2 = tf.contrib.layers.conv2d(relu1, num_outputs=depth2, kernel_size=patch_size2, stride=1, padding='SAME', trainable=True) # padding='SAME' , ,scope='conv2'
                    batch_norm_2 = tf.contrib.layers.batch_norm(conv2, decay=batch_norm_decay, is_training=phase, updates_collections=tf.GraphKeys.UPDATE_OPS) # scale=True ,scope='bn2'
                    pool2 = tf.contrib.layers.max_pool2d(batch_norm_2, kernel_size=2, stride=2, padding='VALID')
                    relu2 = tf.nn.relu(pool2, 'relu2')

                    shape2 = batch_norm_2.get_shape().as_list()
                    print 'shape2 :', shape2

                with tf.variable_scope('layer3'):
                    conv3 = tf.contrib.layers.conv2d(relu2, num_outputs=depth3, kernel_size=patch_size3, stride=1, padding='SAME', trainable=True) # padding='SAME' ,scope='conv3'
                    batch_norm_3 = tf.contrib.layers.batch_norm(conv3, decay=batch_norm_decay, is_training=phase, updates_collections=tf.GraphKeys.UPDATE_OPS) # scale=True ,scope='bn3'
                    relu3 = tf.nn.relu(batch_norm_3, 'relu3')

                    shape3 = batch_norm_3.get_shape().as_list()
                    print 'shape3 :', shape3

                with tf.variable_scope('layer4'):
                    conv4 = tf.contrib.layers.conv2d(relu3, num_outputs=depth3, kernel_size=patch_size3, stride=1, padding='SAME', trainable=True) # padding='SAME' ,scope='conv4'
                    batch_norm_4 = tf.contrib.layers.batch_norm(conv4, decay=batch_norm_decay, is_training=phase, updates_collections=tf.GraphKeys.UPDATE_OPS) # scale=True ,scope='bn4'
                    relu4 = tf.nn.relu(batch_norm_4, 'relu4')

                    shape4 = batch_norm_4.get_shape().as_list()
                    print 'shape4 :', shape4

                with tf.variable_scope('layer5'):
                    conv5 = tf.contrib.layers.conv2d(relu4, num_outputs=depth2, kernel_size=patch_size3, stride=2, padding='VALID', trainable=True) # padding='SAME' ,scope='conv5'
                    batch_norm_5 = tf.contrib.layers.batch_norm(conv5, decay=batch_norm_decay, is_training=phase, updates_collections=tf.GraphKeys.UPDATE_OPS) # scale=True ,scope='bn5'
                    relu5 = tf.nn.relu(batch_norm_5, 'relu5')

                    shape5 = batch_norm_5.get_shape().as_list()
                    print 'shape5 :', shape5

                with tf.variable_scope('fc_layer1'):
                    pool5_flat = tf.reshape(relu5, [shape5[0], shape5[1]*shape5[2]*shape5[3]]) # [batch_size, features]
                    fully_connected_1 = tf.contrib.layers.fully_connected(pool5_flat, num_hidden, activation_fn=tf.nn.relu, trainable=True) #  ,scope='fc1'
                    dropped = tf.contrib.layers.dropout(fully_connected_1, keep_prob=0.5, noise_shape=None)

                    shape6 = fully_connected_1.get_shape().as_list()
                    print 'shape6 :', shape6

                with tf.variable_scope('fc_layer2'):
                    fully_connected_2 = tf.contrib.layers.fully_connected(dropped, num_hidden2, activation_fn=tf.nn.relu, trainable=True) # ,scope='fc2'
                    fc2_right_shape = tf.reshape(fully_connected_2, [batch_size, 55, 74, 1])

                    shape7 = fully_connected_2.get_shape().as_list()
                    print 'shape7 :', shape7
                    right_shape7 = fc2_right_shape.get_shape().as_list()
                    print 'right_shape7 :', right_shape7


            with tf.variable_scope('Fine'):
                with tf.variable_scope('Fine1'):
                    conv_fs_1 = tf.contrib.layers.conv2d(data, num_outputs=fine1depth, kernel_size=fine1_patch_size, stride=2, padding='VALID', trainable=True)#,trainable = finescalebool
                    # bn_fs_1 = tf.contrib.layers.batch_norm(conv_fs_1, decay=batch_norm_decay, is_training=phase, updates_collections=tf.GraphKeys.UPDATE_OPS)#,trainable = finescalebool
                    bn_fs_1 = conv_fs_1
                    pool_fs_1 = tf.contrib.layers.max_pool2d(bn_fs_1, kernel_size=2, stride=2, padding='VALID')
                    relu_fs_1 = tf.nn.relu(pool_fs_1, 'relu_fs1')

                    shape_fs_1 = relu_fs_1.get_shape().as_list()
                    print 'shape_fs1 :',shape_fs_1 

                with tf.variable_scope('Fine2'):
                    concat_fs_2 = tf.nn.relu(tf.concat([relu_fs_1, fc2_right_shape],3),'concat_fs2')

                    shape_fs_2 = concat_fs_2.get_shape().as_list()
                    print 'shape_fs2 :',shape_fs_2

                with tf.variable_scope('Fine3'):
                    conv_fs_3 = tf.contrib.layers.conv2d(concat_fs_2, num_outputs=fine3depth, kernel_size=fine3_patch_size, trainable=True)#,trainable = finescalebool
                    # bn_fs_3 = tf.contrib.layers.batch_norm(conv_fs_3, decay=batch_norm_decay, is_training=phase, updates_collections=tf.GraphKeys.UPDATE_OPS)#,trainable = finescalebool
                    bn_fs_3 = conv_fs_3
                    relu_fs_3 = tf.nn.relu(bn_fs_3, 'relu_fs3')

                    shape_fs_3 = relu_fs_3.get_shape().as_list()
                    print 'shape_fs3 :',shape_fs_3 

                with tf.variable_scope('Fine4'):
                    conv_fs_4 = tf.contrib.layers.conv2d(relu_fs_3, num_outputs=fine4depth, kernel_size=fine4_patch_size, trainable=True) #,trainable = finescalebool

                    shape_fs_4 = conv_fs_4.get_shape().as_list()
                    print 'shape_fs4 :',shape_fs_4 

            return fully_connected_2, fc2_right_shape, conv_fs_4

        # Training
        depth_map, depth_map_right_shape, depth_map_fine = model(tf_train_dataset, phase)
        ground_truth = tf.reshape(tf_train_labels, [batch_size, 55*74])
        depth_map_fs = tf.reshape(depth_map_fine, [batch_size, 55*74])

        # loss = tf.nn.l2_loss(tf.subtract(depth_map, ground_truth))
        loss = scale_invariant_loss(depth_map,ground_truth,Hout*Wout) # *batch_size
        # loss = RMSEf_inv(depth_map,ground_truth,Hout*Wout*batch_size)
        fine_loss = scale_invariant_loss(depth_map_fs,ground_truth,Hout*Wout*batch_size)

        image = depth_map_right_shape
        image_gt = tf.reshape(tf_train_labels, [batch_size,55,74,1])
        fine_image = depth_map_fine

        depthimg_abs_fs = create_depth_image(depth_map_fs, ground_truth)
        fine_image_corrected = tf.reshape(depthimg_abs_fs, [batch_size,55,74,1])
        RMSElog = RMSE_log_inv(depthimg_abs_fs, ground_truth, Hout*Wout*batch_size)
        RMSElinear = RMSE_linear(depthimg_abs_fs, ground_truth, Hout*Wout*batch_size)
        tresholderror = treshold(depthimg_abs_fs, ground_truth, Hout*Wout*batch_size)

        variables_coarse = [v for v in tf.trainable_variables() if not 'Fine' in v.name]
        variables_fine = [v for v in tf.trainable_variables() if 'Fine' in v.name]

        # print_op = tf.get_variable('Coarse/layer1/Conv/weights:0')
        # print 'Coarse ______________________________________________'
        # for k in variables_coarse:
        #     print "Variable: ", k
        # print 'Fine ______________________________________________'
        # for k in variables_fine:
        #     print "Variable: ", k

        # Optimizer
        # Ensure that we execute the update_ops before perfomring the train step
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss, var_list=variables_coarse, global_step=global_step) # tf.train.AdamOptimizer  # var_list=variables_coarse,
            optimizer_fs = tf.train.AdadeltaOptimizer(learning_rate).minimize(fine_loss, var_list=variables_fine, global_step=global_step)
            optimizer_all = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss + fine_loss, global_step=global_step)

        # summary
        loss_sum = tf.summary.scalar('loss', loss)
        loss_valid_sum = tf.summary.scalar('validation_loss', loss)
        img_sum = tf.summary.image('2img', image, max_outputs = 3)
        img_sum_2 = tf.summary.image('1img_gt', image_gt, max_outputs = 3) # max_outputs = batch_size
        img_sum_val = tf.summary.image('5img_val', image, max_outputs = 3)
        img_sum_val_2 = tf.summary.image('4img_val_gt', image_gt, max_outputs = 3)

        fs_loss_sum = tf.summary.scalar('fine-scale loss', fine_loss)
        fs_loss_valid_sum = tf.summary.scalar('fine-scale validation_loss', fine_loss)
        fs_img_sum = tf.summary.image('3fine-scale img', fine_image_corrected, max_outputs = 3)
        fs_img_sum_val = tf.summary.image('6fine-scale img_val', fine_image_corrected, max_outputs = 3)

        # merge summaries into single operation
        summary_op = tf.summary.merge([loss_sum, fs_loss_sum])
        summary_op_2 = tf.summary.merge([loss_valid_sum, fs_loss_valid_sum])
        summary_op_3 = tf.summary.merge([img_sum, img_sum_2, fs_img_sum]) #loss_sum
        summary_op_4 = tf.summary.merge([img_sum_val, img_sum_val_2, fs_img_sum_val])

        

    gpu_options = tf.GPUOptions(allow_growth=True)
    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
    with tf.Session(graph=graph, config=tf_config) as session:
        gl_vars = [v.name for v in tf.global_variables()]
        for k in gl_vars:
            print "Variable: ", k

        print 'Coarse ______________________________________________'
        for k in variables_coarse:
            print "Variable: ", k
        saver_start = tf.train.Saver(max_to_keep=3, var_list=variables_coarse)

        # Create log writer
        writer = tf.summary.FileWriter(logdir, session.graph)
        # Create a saver
        saver = tf.train.Saver(max_to_keep=3)

        # Check for existing checkpoints
        ckpt = tf.train.latest_checkpoint(checkpoint)

        if ckpt == None:
            # Initialize Variables
            tf.global_variables_initializer().run()
            print 'no checkpoint found' 
            num_epochs_start = 0
        else:
            saver.restore(session,ckpt)
            print 'checkpoint restored'
            num_epochs_start = int(session.run(global_step)/number_of_batches)

        number_of_batches_start = int(session.run(global_step)) % number_of_batches

        print 'Initialized'

        batch_data = None
        batch_labels = None

        # if fsbool == False:
        #     coarse_train = 1
        #     fine_train = 1
        # else:
        #     coarse_train = 1
        #     fine_train = 1
        if trainbool == False:
            num_epochs = 1
            num_epochs_start = 0
            number_of_batches_start = 0

        # for key in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES): print(key) 
        # trainable_vars = [v.name for v in tf.trainable_variables()]
        # for k in trainable_vars:
        #     print "Variable: ", k
        print 'number epoch start' , num_epochs_start
        print 'number epock end' , num_epochs 
        for epoch in range(num_epochs_start,num_epochs): 
            # dataGenerator.shuffle(number_of_val_batches*batch_size, number_of_batches*batch_size)
            # for i in range(1,10):
            #     print dataGenerator.randomList[i]
            
            # print "EPOCH: %d" % epoch
            test_value_avg = 0
            RMSE_l_i_avg = 0
            RMSE_l_avg = 0
            time_avg = 0
            print "number_of_batches_start: %d" % number_of_batches_start
            for step in range(number_of_batches_start,number_of_batches):
                # Print step
                print "epoch: %d" % (epoch)
                print "we are at session step: %d  and step: %d" % (session.run(global_step), step)

                # DATA ___________________________________________________________________________________________________________________
                # Get training data
                in_rgb, in_depth =  dataGenerator.__getitem__(step + number_of_val_batches)

                # img_rgb = np.reshape(in_rgb, [batch_size, 228,304,3])
                # img_rgb_1 = img_rgb[0, :, :, :]
                # img_rgb_2 = img_rgb[1, :, :, :]
                # img_rgb_3 = img_rgb[2, :, :, :]
                # img_rgb_1 = im.fromarray(img_rgb_1.astype('uint8'))
                # img_rgb_2 = im.fromarray(img_rgb_2.astype('uint8'))
                # img_rgb_3 = im.fromarray(img_rgb_3.astype('uint8'))

                # plt.figure(1)
                # plt.imshow(img_rgb_1)

                # plt.figure(2)
                # plt.imshow((img_rgb_2))

                # plt.figure(3)
                # plt.imshow((img_rgb_3))

                # plt.show()
                # pdb.set_trace()

                in_rgb = (in_rgb - 127.5) / 127.5
                batch_data = in_rgb
                batch_labels = in_depth
                # Get validation batch
                if trainbool == True:
                    if step % 5 == 0:
                        random_val_batch_number= random.randint(1,number_of_val_batches)
                        val_rgb, val_depth = dataGenerator.__getitem__(random_val_batch_number)
                        val_rgb = (val_rgb - 127.5) / 127.5
                else:
                    val_rgb, val_depth = dataGenerator.__getitem__(step)
                    val_rgb = (val_rgb - 127.5) / 127.5
                    # test_rgb, test_depth = dataGenerator.__getitem__(23000+step) # 23000 batches of size 20 = 460000
                    # test_rgb = (test_rgb - 127.5) / 127.5

                # TESTING ________________________________________________________________________________________________________________

                # Training = false -> Load in model and test on some data, nothing should train
                if trainbool == False:
                    feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, phase: 0}
                    feed_dict_val = {tf_train_dataset: val_rgb, tf_train_labels: val_depth, phase: 0}
                    # feed_dict_test = {tf_train_dataset: test_rgb, tf_train_labels: test_depth, phase: 0}
                    l, RMSE_l_i, RMSE_l, test_value, depth_fine_img, depth_img_coarse  = session.run([loss, RMSElog, RMSElinear, tresholderror, depthimg_abs_fs, image], feed_dict=feed_dict)

                    # start1 = time()
                    # finedepthmap  = session.run([fine_image], feed_dict=feed_dict)
                    # end1 = time()

                    # print 'time elapsed: %f' % (end1-start1)

                    print ' -> RMSEloginv: %f' % (RMSE_l_i)

                    print ' -> RMSElinear: %f' % (RMSE_l)

                    print ' -> tresholderror: %f' % (test_value)

                    print '_________________________________________'

                    RMSE_l_i_avg = RMSE_l_i_avg + (RMSE_l_i - RMSE_l_i_avg) / (step+1)
                    print 'RMSEloginv_avg: %f' % (RMSE_l_i_avg)

                    RMSE_l_avg = RMSE_l_avg + (RMSE_l - RMSE_l_avg) / (step+1)
                    print 'RMSElinear_avg: %f' % (RMSE_l_avg)

                    test_value_avg = test_value_avg + (test_value - test_value_avg) / (step+1)
                    print 'tresholderror_avg: %f' % (test_value_avg)

                    # time_avg = time_avg + ((end1-start1) - time_avg) / (step+1)
                    # print 'time_avg: %f' % (time_avg)

                    # depth_img1 = np.reshape(depth_fine_img, [batch_size, 55,74])
                    # print depth_img1.shape
                    # depth_img1 = depth_img1[7, :, :]

                    # depth_img2 = np.reshape(depth_img_coarse, [batch_size, 55,74])
                    # print depth_img2.shape
                    # depth_img2 = depth_img2[7, :, :]


                    # plt.imsave("/users/start2012/r0298867/Thesis/implementation1/build_new/Fine/depth_coarse.png", depth_img2)
                    # plt.imsave("/users/start2012/r0298867/Thesis/implementation1/build_new/Fine/depth_fine.png", depth_img1)
                    
                    # plt.figure(1)
                    # plt.imshow((depth_img1))

                    # depth_img_label = np.reshape(batch_labels, [batch_size, 55,74])
                    # depth_img_label = depth_img_label[7, :, :]
                    # plt.figure(2)
                    # plt.imshow((depth_img_label))

                    # in_rgb2, in_depth2 =  dataGenerator.__getitem__(step + number_of_val_batches)
                    # print in_rgb2.shape
                    # pdb.set_trace()
                    # depth_img_rgb = np.reshape(in_rgb2, [batch_size, 228,304,3])
                    # depth_img_rgb = in_rgb2[7, :, :, :]
                    # plt.figure(3)
                    # plt.imshow((depth_img_rgb))

                    # cv2.imwrite("/users/start2012/r0298867/Thesis/implementation1/build_new/Fine/testfilename.png", depth_img_rgb)
                    # # plt.show()
                    # pdb.set_trace()

                # TRAINING _______________________________________________________________________________________________________________
                
                # The only thing that needs changing when switching between training coarse or fine-scale is  ----- session.run([optimizer_all] -> session.run([optimizer] or session.run([optimizer_fs] -----
                else:
                    # Normal training
                    feed_dict={tf_train_dataset: batch_data, tf_train_labels: batch_labels, phase: 1}
                    # _ = session.run([optimizer_fs], feed_dict=feed_dict)
                    if fsbool == False:
                        _ = session.run([optimizer_all], feed_dict=feed_dict)
                    else:
                        _ = session.run([optimizer_all], feed_dict=feed_dict)

                    tvar = tf.trainable_variables('Coarse/layer1/Conv/weights:0')
                    tvar2 = tf.trainable_variables('Fine/Fine1/Conv/weights:0')
                    tvar_val, tvar_val2 = session.run([tvar, tvar2])

                    for var, val in zip(tvar, tvar_val):
                        print(var.name, val[0,0,0,0])  # Prints the name of the variable alongside its value.
                    for var, val in zip(tvar2, tvar_val2):
                        print(var.name, val[0,0,0,0]) 
                    


                    # Save loss every 5 steps
                    if step % 5 == 0:
                        # Training data
                        l, summary = session.run([loss, summary_op], 
                            feed_dict={tf_train_dataset: batch_data, tf_train_labels: batch_labels, phase: 0})
                        writer.add_summary(summary, global_step=session.run(global_step))
                        writer.flush()    
                        print 'loss: %f' % (l)
                        # print 'RMSE (log, scale-invariant): %f' % (RMSE)

                        # Validation data
                        val_loss, summary  = session.run([loss, summary_op_2], 
                            feed_dict={tf_train_dataset: val_rgb, tf_train_labels: val_depth, phase: 0})
                        writer.add_summary(summary, global_step=session.run(global_step))
                        writer.flush()
                        print 'validation loss: %f' % (val_loss)
                        # print 'validation RMSE (log, scale-invariant): %f' % (RMSE)

                    # If step is multiple of 100 also save an image to summary and save session
                    if (step % 100 == 0):
                        #Training images
                        depth, summary = session.run([depth_map, summary_op_3], 
                            feed_dict={tf_train_dataset: batch_data, tf_train_labels: batch_labels, phase: 0}) 
                        writer.add_summary(summary, global_step=session.run(global_step))
                        writer.flush()

                        #Validation images
                        depth, summary = session.run([depth_map, summary_op_4], 
                            feed_dict={tf_train_dataset: val_rgb, tf_train_labels: val_depth, phase: 0})
                        writer.add_summary(summary, global_step=session.run(global_step))
                        writer.flush()

                        # Save session every 100 steps
                        saver.save(session, savedir, global_step=session.run(global_step))
                        print "Model saved"
                # print " ---> we are at session step: %d  and step: %d" % (session.run(global_step), step)
            # You can resume training in middle of epoch at batch nr 100 for example, but next epoch should start at 0 again            
            number_of_batches_start = 0
            print 'number_of_batches_start reset to 0'


if __name__ == '__main__':

    [dataGenerator,config_file] = load_data()
    train(dataGenerator,config_file)

