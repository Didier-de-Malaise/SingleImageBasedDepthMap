import tensorflow as tf
import numpy as np
import pdb
import load_NYU_Data as loaddata
import PIL.Image as im
import matplotlib.pyplot as plt
import random
import math
import os

from tensorflow.contrib.layers import convolution2d, batch_norm, max_pool2d, fully_connected
#python fix_graph_2.py --num_batches 20 --num_epochs 1000
tf.flags.DEFINE_string("dir", "/users/start2012/r0298867/Thesis/implementation1/tmp3/", "main directory.")
tf.flags.DEFINE_string("learning_rate", 0.1, "learning rate.")
tf.flags.DEFINE_string("num_epochs", 1000, "number of epochs.")
tf.flags.DEFINE_string("num_batches", 20, "number of batches.")
tf.flags.DEFINE_boolean("fine_scale", True, "Train coarse or fine-scale network.")

FLAGS = tf.flags.FLAGS

# Add fine scale layers, finescalebool not included yet. Test if shapes are correct.

def load_data():
    global train_labels
    global train_dataset

    [dataGenerator, config_file] =  loaddata.main()
    return dataGenerator, config_file

    # train_dataset = np.load('saved_rgb_img_10.npy')
    # train_labels = np.load('saved_depth_map_10.npy')

def scale_invariant_loss(depthmap, groundtruth, pixels):
    lambd = 0.5
    d = (depthmap)-tf.log(groundtruth+1e-10)
    n = pixels
    var1 = tf.reduce_sum(tf.pow(d,2),1,keep_dims=True)
    var2 = tf.pow(tf.reduce_sum(d,1,keep_dims=True),2)
    loss = var1/n - lambd*var2/(n**2)
    loss = tf.reduce_sum(loss)

    return loss

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
    total_batches = dataGenerator.batchAm
    # 20% van batches voor validation
    # number_of_batches = int(math.floor(total_batches * 0.8))
    # number_of_val_batches = int(math.floor(total_batches * 0.2))
    # number_of_batches = int(math.floor(total_batches * 0.005))
    # number_of_val_batches = int(math.floor(total_batches * 0.001))
    number_of_batches = int(FLAGS.num_batches)
    number_of_val_batches= 2000
    fsbool = FLAGS.fine_scale

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
    fine2depth = 64
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
        fs_phase = tf.placeholder(tf.bool, name='phase')
        global_step = tf.Variable(0, name='global_step', trainable=False)

        def model(data, phase):
            with tf.name_scope('layer1'):
                conv1 = tf.contrib.layers.conv2d(data,num_outputs=depth1,kernel_size=patch_size1,stride=4,padding='VALID',scope='conv1')
                batch_norm_1 = tf.contrib.layers.batch_norm(conv1,decay=batch_norm_decay,is_training=phase,scope='bn1') # scale=True
                # batch_norm_1 = conv1
                pool1 = tf.contrib.layers.max_pool2d(batch_norm_1,kernel_size=2,stride=2,padding='VALID')
                relu1 = tf.nn.relu(pool1,'relu1')

                shape1 = batch_norm_1.get_shape().as_list()
                print 'shape1 :',shape1

            with tf.name_scope('layer2'):
                conv2 = tf.contrib.layers.conv2d(relu1,num_outputs=depth2,kernel_size=patch_size2,stride=1,padding='SAME',scope='conv2') # padding='SAME'
                batch_norm_2 = tf.contrib.layers.batch_norm(conv2,decay=batch_norm_decay,is_training=phase,scope='bn2') # scale=True
                # batch_norm_2 = conv2
                pool2 = tf.contrib.layers.max_pool2d(batch_norm_2,kernel_size=2,stride=2,padding='VALID')
                relu2 = tf.nn.relu(pool2,'relu2')

                shape2 = batch_norm_2.get_shape().as_list()
                print 'shape2 :', shape2

            with tf.name_scope('layer3'):
                conv3 = tf.contrib.layers.conv2d(relu2,num_outputs=depth3,kernel_size=patch_size3,stride=1,padding='SAME',scope='conv3') # padding='SAME'
                batch_norm_3 = tf.contrib.layers.batch_norm(conv3,decay=batch_norm_decay,is_training=phase,scope='bn3') # scale=True
                # batch_norm_3 = conv3
                relu3 = tf.nn.relu(batch_norm_3,'relu3')

                shape3 = batch_norm_3.get_shape().as_list()
                print 'shape3 :', shape3

            with tf.name_scope('layer4'):
                conv4 = tf.contrib.layers.conv2d(relu3,num_outputs=depth3,kernel_size=patch_size3,stride=1,padding='SAME',scope='conv4') # padding='SAME'
                batch_norm_4 = tf.contrib.layers.batch_norm(conv4,decay=batch_norm_decay,is_training=phase,scope='bn4') # scale=True
                # batch_norm_4 = conv4
                relu4 = tf.nn.relu(batch_norm_4,'relu4')

                shape4 = batch_norm_4.get_shape().as_list()
                print 'shape4 :', shape4

            with tf.name_scope('layer5'):
                conv5 = tf.contrib.layers.conv2d(relu4,num_outputs=depth2,kernel_size=patch_size3,stride=2,padding='VALID',scope='conv5') # padding='SAME'
                batch_norm_5 = tf.contrib.layers.batch_norm(conv5,decay=batch_norm_decay,is_training=phase,scope='bn5') # scale=True
                # batch_norm_5 = conv5
                relu5 = tf.nn.relu(batch_norm_5,'relu5')

                shape5 = batch_norm_5.get_shape().as_list()
                print 'shape5 :', shape5

            with tf.name_scope('fc_layer1'):
                pool5_flat = tf.reshape(relu5, [shape5[0], shape5[1]*shape5[2]*shape5[3]]) # [batch_size, features]
                fully_connected_1 = tf.contrib.layers.fully_connected(pool5_flat,4096,activation_fn=tf.nn.relu,scope='fc1') # 
                # batch_norm_6 = tf.contrib.layers.batch_norm(fully_connected_1,decay=batch_norm_decay,scale=True,is_training=phase,scope='bn6')
                # batch_norm_6 = fully_connected_1

                shape6 = fully_connected_1.get_shape().as_list()
                print 'shape6 :', shape6

            with tf.name_scope('fc_layer2'):
                fully_connected_2 = tf.contrib.layers.fully_connected(fully_connected_1,4070,activation_fn=tf.nn.relu,scope='fc2')
                fc2_right_shape = tf.reshape(fully_connected_2, [batch_size, 55, 74, 1])

                shape7 = fully_connected_2.get_shape().as_list()
                print 'shape7 :', shape7
                right_shape7 = fc2_right_shape.get_shape().as_list()
                print 'right_shape7 :', right_shape7

            with tf.name_scope('Fine1'):
                conv_fs_1 = tf.contrib.layers.conv2d(data,num_outputs=fine1depth,kernel_size=fine1_patch_size,stride=2,padding='VALID',scope='conv_fs1')#,trainable = finescalebool
                bn_fs_1 = tf.contrib.layers.batch_norm(conv_fs_1,decay=batch_norm_decay,is_training=fs_phase,scope='bn_fs1')#,trainable = finescalebool
                pool_fs_1 = tf.contrib.layers.max_pool2d(bn_fs_1,kernel_size=2,stride=2,padding='VALID')
                relu_fs_1 = tf.nn.relu(pool_fs_1,'relu_fs1')

                shape_fs_1 = relu_fs_1.get_shape().as_list()
                print 'shape_fs1 :',shape_fs_1 

            with tf.name_scope('Fine2'):
                concat_fs_2 = tf.nn.relu(tf.concat([relu_fs_1, fc2_right_shape],3),'concat_fs2')

                shape_fs_2 = concat_fs_2.get_shape().as_list()
                print 'shape_fs2 :',shape_fs_2

            with tf.name_scope('Fine3'):
                conv_fs_3 = tf.contrib.layers.conv2d(concat_fs_2,num_outputs=fine3depth,kernel_size=fine3_patch_size,scope='conv_fs3')#,trainable = finescalebool
                bn_fs_3 = tf.contrib.layers.batch_norm(conv_fs_3,decay=batch_norm_decay,is_training=fs_phase,scope='bn_fs3')#,trainable = finescalebool
                relu_fs_3 = tf.nn.relu(bn_fs_3,'relu_fs3')

                shape_fs_3 = relu_fs_3.get_shape().as_list()
                print 'shape_fs3 :',shape_fs_3 

            with tf.name_scope('Fine4'):
                conv_fs_4 = tf.contrib.layers.conv2d(relu_fs_3,num_outputs=fine4depth,kernel_size=fine4_patch_size,scope='conv_fs4') #,trainable = finescalebool

                shape_fs_4 = conv_fs_4.get_shape().as_list()
                print 'shape_fs4 :',shape_fs_4 

            return fully_connected_2, conv_fs_4

        # Training
        depth_map, depth_map_fine = model(tf_train_dataset,phase)
        ground_truth = tf.reshape(tf_train_labels, [batch_size, 55*74])
        depth_map_fs = tf.reshape(depth_map_fine, [batch_size, 55*74])

        # loss = tf.nn.l2_loss(tf.subtract(depth_map, ground_truth))
        loss = scale_invariant_loss(depth_map,ground_truth,Hout*Wout)
        image = tf.reshape(depth_map, [batch_size, 55, 74, 1])
        image_gt = tf.reshape(tf_train_labels, [batch_size,55,74,1])
        fine_loss = scale_invariant_loss(depth_map_fs,ground_truth,Hout*Wout)
        fine_image = depth_map_fine

        variables_coarse = [v for v in tf.trainable_variables() if not 'fs' in v.name]
        variables_fine = [v for v in tf.trainable_variables() if 'fs' in v.name]

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
            optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss, var_list=variables_coarse, global_step=global_step) # tf.train.AdamOptimizer # var_list=variables_coarse
            optimizer_fs = tf.train.AdadeltaOptimizer(learning_rate).minimize(fine_loss, var_list=variables_fine, global_step=global_step)

        # summary
        loss_sum = tf.summary.scalar('loss', loss)
        loss_valid_sum = tf.summary.scalar('validation_loss', loss)
        img_sum = tf.summary.image('img', image, max_outputs = 3)
        img_sum_2 = tf.summary.image('img_gt', image_gt, max_outputs = 3) # max_outputs = batch_size
        img_sum_val = tf.summary.image('img_val', image, max_outputs = 3)
        img_sum_val_2 = tf.summary.image('img_val_gt', image_gt, max_outputs = 3)

        fs_loss_sum = tf.summary.scalar('fine-scale loss', fine_loss)
        fs_loss_valid_sum = tf.summary.scalar('fine-scale validation_loss', fine_loss)
        fs_img_sum = tf.summary.image('fine-scale img', fine_image, max_outputs = 3)
        fs_img_sum_val = tf.summary.image('fine-scale img_val', fine_image, max_outputs = 3)

        # merge summaries into single operation
        summary_op = tf.summary.merge([loss_sum, fs_loss_sum])
        summary_op_2 = tf.summary.merge([loss_valid_sum, fs_loss_valid_sum])
        summary_op_3 = tf.summary.merge([img_sum, img_sum_2, fs_img_sum]) #loss_sum
        summary_op_4 = tf.summary.merge([img_sum_val, img_sum_val_2, fs_img_sum_val])

    gpu_options = tf.GPUOptions(allow_growth=True)
    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
    with tf.Session(graph=graph, config=tf_config) as session:
        # gl_vars = [v.name for v in tf.global_variables()]
        # for k in gl_vars:
        #     print "Variable: ", k

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

        if number_of_batches >= 20000:
            number_of_batches_start = int(session.run(global_step))
        else:
            number_of_batches_start = 0

        print 'Initialized'

        i = 0
        batch_data = None
        batch_labels = None
        depth = None
        if fsbool == False:
            phase_train = 1
            fs_phase_train = 0
        else:
            phase_train = 0
            fs_phase_train = 1

        # for key in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES): print(key) 
        # trainable_vars = [v.name for v in tf.trainable_variables()]
        # for k in trainable_vars:
        #     print "Variable: ", k

        for epoch in range(num_epochs_start,num_epochs): 
            # dataGenerator.shuffle(number_of_val_batches*batch_size, number_of_batches*batch_size)
            # for i in range(1,10):
            #     print dataGenerator.randomList[i]
            print "EPOCH: %d" % epoch    

            # --------------------------------------------------------------------------------

            # This should be test data, validation should happen after every batch or step 
            # (what i called step earlier is an epoch)
            # That is the goal of using batches, you calculate gradient/update parameters after each step
            # Calculate validation loss after each step on a random batch

            # Look at tensorboard to interpret the loss functions and adjust hyper parameters

            # --------------------------------------------------------------------------------

            for step in range(number_of_batches_start,number_of_batches):
                # Print step
                print "we are at step: %d" % step

                # Get training data
                in_rgb, in_depth =  dataGenerator.__getitem__(step + number_of_val_batches)
                in_rgb = (in_rgb - 127.5) / 127.5
                batch_data = in_rgb
                batch_labels = in_depth

                if step % 5 == 0:
                    random_val_batch_number= random.randint(1,number_of_val_batches)
                    val_rgb, val_depth = dataGenerator.__getitem__(random_val_batch_number)
                    val_rgb = (val_rgb - 127.5) / 127.5
                
                # Don't forget to put phase on 0 when training fine-scale network !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                if step % 5 == 0:
                    val_loss, valid_depth, summary = session.run([loss, depth_map, summary_op_2], 
                        feed_dict={tf_train_dataset: val_rgb, tf_train_labels: val_depth, phase: 0, fs_phase: 0})
                    writer.add_summary(summary, global_step=session.run(global_step))
                    writer.flush()
                    print 'validation loss: %f, step: %d' % (val_loss, step)

                    _, _, l, depth, summary = session.run([optimizer, optimizer_fs, loss, depth_map, summary_op], 
                        feed_dict={tf_train_dataset: batch_data, tf_train_labels: batch_labels, phase: 1, fs_phase: 1})
                    writer.add_summary(summary, global_step=session.run(global_step))
                    writer.flush()    
                    print 'loss: %f, step: %d, epoch: %d' % (l, step, epoch)

                else:
                    feed_dict={tf_train_dataset: batch_data, tf_train_labels: batch_labels, phase: 1, fs_phase: 1}
                    _, _ = session.run([optimizer, optimizer_fs], feed_dict=feed_dict)

                # If step is multiple of 100 also save an image to summary and save session
                if step % 100 == 0:
                    #Training images
                    depth, summary = session.run([depth_map, summary_op_3], 
                        feed_dict={tf_train_dataset: batch_data, tf_train_labels: batch_labels, phase: 1, fs_phase: 1})
                    writer.add_summary(summary, global_step=session.run(global_step))
                    writer.flush()

                    #Validation images
                    depth, summary = session.run([depth_map, summary_op_4], 
                        feed_dict={tf_train_dataset: val_rgb, tf_train_labels: val_depth, phase: 0, fs_phase: 0})
                    writer.add_summary(summary, global_step=session.run(global_step))
                    writer.flush()

                    # Save session every 100 steps
                    saver.save(session, savedir, global_step=session.run(global_step))
                    print "Model saved"



if __name__ == '__main__':

    [dataGenerator,config_file] = load_data()
    train(dataGenerator,config_file)

