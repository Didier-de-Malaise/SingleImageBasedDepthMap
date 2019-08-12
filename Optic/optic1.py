import tensorflow as tf
import numpy as np
import pdb
import load_NYU_Data as loaddata
import load_KITTY_Data as kittydata
import load_Test_Data as testdata 
import PIL.Image as im
import matplotlib.pyplot as plt
import random
import math
import os

# from test_data_loader import *
from transformer import *
from visualize import *
from tensorflow.contrib.layers import convolution2d, batch_norm, max_pool2d, fully_connected

tf.flags.DEFINE_string("dir", "/users/start2012/r0298867/Thesis/implementation1/tmp/", "main directory.")
# tf.flags.DEFINE_string("dir", "/esat/opal/r0298867/r0298867/runs/optic/5Optic_1image_grijze_vlekken/", "main directory.")
tf.flags.DEFINE_float("learning_rate", 0.01, "learning rate.")
tf.flags.DEFINE_integer("num_epochs", 1101, "number of epochs.")
tf.flags.DEFINE_integer("num_batches", 1, "number of batches.")
tf.flags.DEFINE_integer("img_output_amount", 3, "amount of images displayed in tensorboard.")
tf.flags.DEFINE_boolean("kitti", True, "Use kitty dataset if boolean is true")
tf.flags.DEFINE_boolean("train", True, "Train network or evaluate when false")

FLAGS = tf.flags.FLAGS


def load_data():

    kitty_data = FLAGS.kitti
    if kitty_data == False:
        [dataGenerator, config_file] =  loaddata.main()
    else:
        [dataGenerator, config_file] = kittydata.main()
    return dataGenerator, config_file

def train(dataGenerator, config_file):
    learning_rate = float(FLAGS.learning_rate)
    batch_norm_decay = 0.99
    num_epochs = int(FLAGS.num_epochs)
    max_out = int(FLAGS.img_output_amount)
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
    number_of_batches = int(FLAGS.num_batches)
    # Number of val batches needs to be even, because every uneven batch corresponds to the next images compared to the previous (even) batch
    number_of_val_batches= 10
    trainbool = FLAGS.train

    # Conv filter size & kernel size
    patch_size1 = 7
    patch_size2 = 5
    patch_size3 = 3
    depth1 = 64
    depth2 = 128
    depth3 = 256
    depth4 = 512
    depth5 = 1024
    depthout = 2
    upsize1 = [114,152]
    upsize2 = [228,304]

    logdir = '%sgraph' % FLAGS.dir
    savedir = '%ssaver/test' % FLAGS.dir
    checkpoint = '%ssaver/' % FLAGS.dir

    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)

    graph = tf.Graph()
    with graph.as_default():        
        input1 = tf.placeholder(tf.float32, shape=(batch_size, 224, 224, 3), name='input1')
        input2 = tf.placeholder(tf.float32, shape=(batch_size, 224, 224, 3), name='input2')
        phase = tf.placeholder(tf.bool, name='phase')

        global_step = tf.Variable(0, name='global_step', trainable=False)

        def model(input1, input2):
            input_concat = tf.concat([input1, input2],3)
            shape_input = input_concat.get_shape().as_list()
            print 'shape_input :', shape_input

            with tf.variable_scope('convolution_layers'):


                conv1 = tf.contrib.layers.conv2d(input_concat, num_outputs=depth1, kernel_size=patch_size1, stride=2, padding='SAME', scope='conv1')
                batch_norm_1 = tf.contrib.layers.batch_norm(conv1, decay=batch_norm_decay, is_training=phase, scope='bn1')
                batch_norm_1 = conv1
                relu1 = tf.nn.relu(batch_norm_1, 'relu1')
                shape1 = batch_norm_1.get_shape().as_list()
                print 'shape1 :',shape1

                conv2 = tf.contrib.layers.conv2d(relu1, num_outputs=depth2, kernel_size=patch_size2, stride=2, padding='SAME', scope='conv2')
                batch_norm_2 = tf.contrib.layers.batch_norm(conv2, decay=batch_norm_decay, is_training=phase, scope='bn2')
                batch_norm_2 = conv2
                relu2 = tf.nn.relu(batch_norm_2, 'relu2')
                shape2 = batch_norm_2.get_shape().as_list()
                print 'shape2 :',shape2

                conv3 = tf.contrib.layers.conv2d(relu2, num_outputs=depth3, kernel_size=patch_size3, stride=2, padding='SAME', scope='conv3')
                batch_norm_3 = tf.contrib.layers.batch_norm(conv3, decay=batch_norm_decay, is_training=phase, scope='bn3')
                batch_norm_3 = conv3
                relu3 = tf.nn.relu(batch_norm_3, 'relu3')
                shape3 = batch_norm_3.get_shape().as_list()
                print 'shape3 :',shape3

                conv4 = tf.contrib.layers.conv2d(relu3, num_outputs=depth3, kernel_size=patch_size3, stride=1, padding='SAME', scope='conv4')
                batch_norm_4 = tf.contrib.layers.batch_norm(conv4, decay=batch_norm_decay, is_training=phase, scope='bn4')
                batch_norm_4 = conv4
                relu4 = tf.nn.relu(batch_norm_4, 'relu4')
                shape4 = batch_norm_4.get_shape().as_list()
                print 'shape4 :',shape4

                conv5 = tf.contrib.layers.conv2d(relu4, num_outputs=depth4, kernel_size=patch_size3, stride=2, padding='SAME', scope='conv5')
                batch_norm_5 = tf.contrib.layers.batch_norm(conv5, decay=batch_norm_decay, is_training=phase, scope='bn5')
                batch_norm_5 = conv5
                relu5 = tf.nn.relu(batch_norm_5, 'relu5')
                shape5 = batch_norm_5.get_shape().as_list()
                print 'shape5 :',shape5

                conv6 = tf.contrib.layers.conv2d(relu5, num_outputs=depth4, kernel_size=patch_size3, stride=1, padding='SAME', scope='conv6')
                batch_norm_6 = tf.contrib.layers.batch_norm(conv6, decay=batch_norm_decay, is_training=phase, scope='bn6')
                batch_norm_6 = conv6
                relu6 = tf.nn.relu(batch_norm_6, 'relu6')
                shape6 = batch_norm_6.get_shape().as_list()
                print 'shape6 :',shape6

                conv7 = tf.contrib.layers.conv2d(relu6, num_outputs=depth5, kernel_size=patch_size3, stride=2, padding='SAME', scope='conv7')
                batch_norm_7 = tf.contrib.layers.batch_norm(conv7, decay=batch_norm_decay, is_training=phase, scope='bn7')
                batch_norm_7 = conv7
                relu7 = tf.nn.relu(batch_norm_7, 'relu7')
                shape7 = batch_norm_7.get_shape().as_list()
                print 'shape7 :',shape7

                
            with tf.variable_scope('deconvolution_layers'):

                predict_flow7 = tf.contrib.layers.conv2d(relu7, num_outputs=2, kernel_size=patch_size3, stride=1, padding='SAME', scope='conv_pf7')
                upsample_pf7 = tf.contrib.layers.conv2d_transpose(predict_flow7, num_outputs=2, kernel_size=4, stride=2, padding='SAME')
                deconv7_6 = tf.contrib.layers.conv2d_transpose(relu7, num_outputs=512, kernel_size=4, stride=2, padding='SAME')

                upsample_pf7 = tf.image.resize_images(upsample_pf7, [14,14])
                # deconv7_6 = tf.image.resize_images(deconv7_6, [13,18])

                shape8 = predict_flow7.get_shape().as_list()
                print 'predict_flow7 :',shape8
                shape9 = upsample_pf7.get_shape().as_list()
                print 'upsample_pf7 :',shape9
                shape10 = deconv7_6.get_shape().as_list()
                print 'deconv7_6 :',shape10


                concat1 = tf.concat([relu6, deconv7_6, upsample_pf7], 3)
                shape11 = concat1.get_shape().as_list()
                print 'concat1 :',shape11


                predict_flow6 = tf.contrib.layers.conv2d(concat1, num_outputs=2, kernel_size=patch_size3, stride=1, padding='SAME', scope='conv_pf6')
                upsample_pf6 = tf.contrib.layers.conv2d_transpose(predict_flow6, num_outputs=2, kernel_size=4, stride=2, padding='SAME')
                deconv6_4 = tf.contrib.layers.conv2d_transpose(concat1, num_outputs=256, kernel_size=4, stride=2, padding='SAME')

                upsample_pf6 = tf.image.resize_images(upsample_pf6, [28,28])
                # deconv6_4 = tf.image.resize_images(deconv6_4, [28,37])

                shape12 = predict_flow6.get_shape().as_list()
                print 'predict_flow6 :',shape12
                shape13 = upsample_pf6.get_shape().as_list()
                print 'upsample_pf6 :',shape13
                shape14 = deconv6_4.get_shape().as_list()
                print 'deconv6_4 :',shape14

                concat2 = tf.concat([relu4, deconv6_4, upsample_pf6], 3)
                shape15 = concat2.get_shape().as_list()
                print 'concat2 :',shape15



                predict_flow4 = tf.contrib.layers.conv2d(concat2, num_outputs=2, kernel_size=patch_size3, stride=1, padding='SAME', scope='conv_pf4')
                upsample_pf4 = tf.contrib.layers.conv2d_transpose(predict_flow4, num_outputs=2, kernel_size=4, stride=2, padding='SAME')
                deconv4_2 = tf.contrib.layers.conv2d_transpose(concat2, num_outputs=128, kernel_size=4, stride=2, padding='SAME')

                upsample_pf4 = tf.image.resize_images(upsample_pf4, [56,56])
                # deconv4_2 = tf.image.resize_images(deconv4_2, [57,76])

                shape16 = predict_flow4.get_shape().as_list()
                print 'predict_flow4 :',shape16
                shape17 = upsample_pf4.get_shape().as_list()
                print 'upsample_pf4 :',shape17
                shape18 = deconv4_2.get_shape().as_list()
                print 'deconv4_2 :',shape18

                concat3 = tf.concat([relu2, deconv4_2, upsample_pf4], 3)
                shape19 = concat3.get_shape().as_list()
                print 'concat2 :',shape19

                predict_flow = tf.contrib.layers.conv2d(concat3, num_outputs=2, kernel_size=patch_size3, stride=1, padding='SAME', scope='conv_pf')
                output = tf.image.resize_images(predict_flow, [224,224])
                shape20 = output.get_shape().as_list()
                print 'output :',shape20

            return output

        # Training
        optic = model(input1, input2)
        # visual_flow = tf.py_func(visualize,[optic], tf.float32)
        # pdb.set_trace()
        # visualoptic = visualize(optic,batch_size,image_size_x,image_size_y)

        prediction_input2 = bilinear_sampler_1d_h(input1, optic)
        difference = tf.pow(tf.abs(prediction_input2 - input2),2)
        difference_reduced = tf.reduce_sum(difference) / (Hout*Wout*batch_size)
        loss = difference_reduced
        # loss = tf.reduce_sum(difference_reduced * optic[:,:,:,0] + difference_reduced * optic[:,:,:,1]) + tf.reduce_sum(difference)
        
        # Optimizer
        # Ensure that we execute the update_ops before perfomring the train step
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdadeltaOptimizer(learning_rate) #.minimize(loss, global_step=global_step)
            # apply_gradient_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        grads = optimizer.compute_gradients(loss)
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

        # summary
        loss_sum = tf.summary.scalar('loss', loss)
        loss_valid_sum = tf.summary.scalar('validation_loss', loss)

        img_sum_1 = tf.summary.image('img2_target', input2, max_outputs = max_out)
        img_sum_2 = tf.summary.image('img2_prediction', prediction_input2, max_outputs = max_out)
        img_sum_3 = tf.summary.image('difference', difference, max_outputs = max_out)

        img_sum_1v = tf.summary.image('v_img2_target', input2, max_outputs = max_out)
        img_sum_2v = tf.summary.image('v_img2_prediction', prediction_input2, max_outputs = max_out)
        img_sum_3v = tf.summary.image('v_difference', difference, max_outputs = max_out)

        # img_sum_4 = tf.summary.image('visual_optic_flow', visual_flow, max_outputs = max_out)

        # img_sum_4 = tf.summary.image('visualize', visual_flow, max_outputs = max_out)
        # img_pred_sum = tf.summary.image('img2_prediction', prediction_input2, max_outputs = 3)

        # merge summaries into single operation
        summary_op_1 = tf.summary.merge([loss_sum])
        summary_op_2 = tf.summary.merge([loss_valid_sum])
        summary_op_3 = tf.summary.merge([img_sum_1, img_sum_2, img_sum_3]) #img_pred_sum  ,img_sum_4
        summary_op_4 = tf.summary.merge([img_sum_1v, img_sum_2v, img_sum_3v]) #img_pred_sum  ,img_sum_4
        # summary_hist = tf.summary.histogram("weights1",kernel)

    gpu_options = tf.GPUOptions(allow_growth=True)
    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
    with tf.Session(graph=graph, config=tf_config) as session:
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

        if trainbool == False:
            num_epochs_start = 0
            num_epochs = 1
            number_of_batches_start = 0

        # gl_vars = [v.name for v in tf.global_variables()]
        # for k in gl_vars:
        #     print "Variable: ", k

        # for key in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES): print(key) 
        # trainable_vars = [v.name for v in tf.trainable_variables()]
        # for k in trainable_vars:
        #     print "Variable: ", k


        # _________________________________________________________________________________________________
        # _________________________________________________________________________________________________
        # ONLY 2 IMAGES
        # _________________________________________________________________________________________________
        # _________________________________________________________________________________________________
        # rgb1 = np.ones((batch_size, image_size_x, image_size_y, 3))
        # f = open('/users/start2012/r0298867/Thesis/Templates/data/image_01/0000000000.png', 'rb')
        # # resize images to 228*304
        # pilIM = im.open(f)
        # new_height = 228
        # wpercent = new_height / float(pilIM.size[1])
        # new_width = int((float(pilIM.size[0]) * float(wpercent)))
        # pilIM = pilIM.resize((new_width, new_height))
        # pilIM = pilIM.crop((0,0,304,228))
        # pilIm2 = pilIM.copy()  # PIL bug workaround
        # f.close()
        # rgb1[i, :, :, :] = np.asarray(pilIM)
        # pilIM.close()

        # rgb2 = np.ones((batch_size, image_size_x, image_size_y, 3))
        # f = open('/users/start2012/r0298867/Thesis/Templates/data/image_01/0000000001.png', 'rb')
        # # resize images to 228*304
        # pilIM = im.open(f)
        # new_height = 228
        # wpercent = new_height / float(pilIM.size[1])
        # new_width = int((float(pilIM.size[0]) * float(wpercent)))
        # pilIM = pilIM.resize((new_width, new_height))
        # pilIM = pilIM.crop((0,0,304,228))
        # pilIm2 = pilIM.copy()  # PIL bug workaround
        # f.close()
        # rgb2[i, :, :, :] = np.asarray(pilIM)
        # pilIM.close()

        # rgb1 = (rgb1 - 127.5) / 127.5
        # rgb2 = (rgb2 - 127.5) / 127.5
        # batch_data1 = rgb1
        # batch_data2 = rgb2
        # print 'test if values of rgb are between -1 and 1' # OK, tested,  min -1, max 1, mean -0.11
        # pdb.set_trace()
        # _________________________________________________________________________________________________
        # _________________________________________________________________________________________________
        

        for epoch in range(num_epochs_start,num_epochs): 
            print "EPOCH: %d" % epoch

            
            if FLAGS.kitti == False:
                range_var = range(number_of_batches_start,number_of_batches,2)
            else:
                range_var = range(number_of_batches_start,number_of_batches)
                    
            for step in range_var:
                print "we are at step: %d" % step

                # NYU_optic hdf5 dataset --------
                # Load 2 batches, only rgb images, depth is not set up in this hdf5 file
                # Batch 1 contains first image, batch 2 contains second image

                # Get training data
                if FLAGS.kitti == False:
                    in_rgb1 =  dataGenerator.__getitem__(step + number_of_val_batches)
                    in_rgb1 = (in_rgb1 - 127.5) / 127.5
                    batch_data1 = in_rgb1

                    in_rgb2 =  dataGenerator.__getitem__(step + 1 + number_of_val_batches)
                    in_rgb2 = (in_rgb2 - 127.5) / 127.5
                    batch_data2 = in_rgb2
                else:
                    in_rgb1, in_rgb2 =  dataGenerator.__getitem__(step + number_of_val_batches)
                    in_rgb1 = (in_rgb1 - 127.5) / 127.5
                    in_rgb2 = (in_rgb2 - 127.5) / 127.5
                    batch_data1 = in_rgb1
                    batch_data2 = in_rgb2

                # # Test to see if images from batch 1 and 2 correspond with each other
                # img1 = np.array(in_rgb1[0,:,:,:])
                # img2 = np.array(in_rgb2[0,:,:,:])
                # plt.figure(1)
                # plt.imshow((img1))
                # plt.figure(2)
                # plt.imshow((img2))
                # plt.show()
                # pdb.set_trace()

                # Get validation batch
                if step % 5 == 0:
                    # NYU Optic ----------
                    if FLAGS.kitti == False:
                        random_val_batch_number = random.randint(1,number_of_val_batches-2)
                        if random_val_batch_number % 2 != 0:
                            random_val_batch_number = random_val_batch_number + 1
                        val_rgb1 = dataGenerator.__getitem__(random_val_batch_number)
                        val_rgb1 = (val_rgb1 - 127.5) / 127.5
                        val_rgb2 = dataGenerator.__getitem__(random_val_batch_number+1)
                        val_rgb2 = (val_rgb2 - 127.5) / 127.5
                    # KITTY Dataset -------------
                    else:
                        random_val_batch_number = random.randint(1,number_of_val_batches-1)
                        val_rgb1, val_rgb2 = dataGenerator.__getitem__(random_val_batch_number)
                        val_rgb1 = (val_rgb1 - 127.5) / 127.5
                        val_rgb2 = (val_rgb2 - 127.5) / 127.5

                # Training = false -> Load in model and test on some data, nothing should train
                if trainbool == False:
                    # Need to load test set in here. For now its just test on trainingsdata.
                    [testdatagenerator, config_file] =  testdata.main()
                    rgb_1, rgb_2, gt = testdatagenerator.__getitem__(step)
                    rgb_1 = (rgb_1 - 127.5) / 127.5
                    rgb_2 = (rgb_2 - 127.5) / 127.5
                    gt = gt / 512

                    l, opticfl, summary, summary_imgs  = session.run([loss, optic, summary_op_1,summary_op_3], 
                        feed_dict={input1: rgb_1, input2: rgb_2, phase: 0})
                    writer.add_summary(summary, global_step=session.run(global_step))
                    writer.add_summary(summary_imgs, global_step=session.run(global_step))
                    writer.flush()
                    print 'loss: %f ' % (l)
                    visualize(opticfl,batch_size,224,224, filename='/users/start2012/r0298867/Thesis/implementation1/build_new/Optic/visualization_test.bmp')

                    print 'rmserror: ' + str(np.sqrt(np.power((np.sum(opticfl) - np.sum(gt[:,:,:,0:1])),2)/(224*224*10)))
                    print 'average endpoint error: ' + str(np.sum(np.sqrt(np.power(opticfl[:,:,:,0] - gt[:,:,:,0],2) + np.power(opticfl[:,:,:,1] - gt[:,:,:,1],2)))/(224*224*10))


                else:
                    # Normal training
                    _, l, opticfl, summary = session.run([apply_gradient_op, loss, optic, summary_op_1], #optimizer
                        feed_dict={input1: batch_data1, input2: batch_data2, phase: 1})
                    writer.add_summary(summary, global_step=session.run(global_step))
                    print 'loss: %f, step: %d, epoch: %d' % (l, step, epoch)
                    print 'min opticflow: %f, max opticflow: %f' % (np.min(opticfl), np.max(opticfl))
                    print 'mean: %f' % (np.mean(opticfl))
                    # Save validation loss every 5 steps 
                    if step % 5 == 0:
                        val_loss, valid_opticfl, summary = session.run([loss, optic, summary_op_2], 
                            feed_dict={input1: val_rgb1, input2: val_rgb2, phase: 0})
                        writer.add_summary(summary, global_step=session.run(global_step))
                        writer.flush()
                        print 'validation loss: %f, step: %d' % (val_loss, step)

                    # If step is multiple of 100 also save an image to summary and save session
                    if step % 100 == 0:
                        #Training images (Opticflow has size [1,228,304,2])
                        opticfl, summary = session.run([optic, summary_op_3], 
                            feed_dict={input1: batch_data1, input2: batch_data2, phase: 0})
                        writer.add_summary(summary, global_step=session.run(global_step))
                        writer.flush()
                        visualize(opticfl,batch_size,224,224)
                        print 'mean: ' , np.mean(opticfl)
                        print 'var: ' , np.var(opticfl)
                        # Why do I track mean and variance again? -> To see if values are close to each other (var) and not too big (mean)
                        # Technically I want a value between 0 and 1, with 1 being the a pixel moving the length of the picture

                        #Validation images
                        valid_opticfl, summary = session.run([optic, summary_op_4], 
                            feed_dict={input1: val_rgb1, input2: val_rgb2, phase: 0})
                        writer.add_summary(summary, global_step=session.run(global_step))
                        writer.flush()

                        # Save session every 100 steps
                        saver.save(session, savedir, global_step=session.run(global_step))
                        print "Model saved"

            number_of_batches_start = 0
            print 'number_of_batches_start reset to 0'



if __name__ == '__main__':

    [dataGenerator,config_file] = load_data()
    train(dataGenerator,config_file)

