import tensorflow as tf


class Model(object):
    def __init__(self, batch_size, no_of_epochs, learning_rate):
        self.no_of_classes = 2
        self.set_hyper_parameters(batch_size, no_of_epochs, learning_rate)

    def set_hyper_parameters(self, batch_size, no_of_epochs, learning_rate):
        self.batch_size = batch_size
        self.no_of_epochs = no_of_epochs
        self.learning_rate = learning_rate

    def optimize(self, decoder_output, correct_label):
        self.logits = tf.reshape(decoder_output, (-1, self.no_of_classes))
        correct_label = tf.reshape(correct_label, (-1, self.no_of_classes))

        self.cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=correct_label))

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(self.cross_entropy_loss)

    def create_model_graph(self, sess, vgg_path, correct_label):
        input_image, keep_prob, vgg_layer3, vgg_layer4, vgg_layer7 = Model.create_encoder(sess, vgg_path)
        last_layer = self.create_decoder(vgg_layer3, vgg_layer4, vgg_layer7)
        self.optimize(last_layer, correct_label)
        return input_image, keep_prob, last_layer

    @staticmethod
    def create_encoder(sess, vgg_path):
        vgg_tag = 'vgg16'
        vgg_input_tensor_name = 'image_input:0'
        vgg_keep_prob_tensor_name = 'keep_prob:0'
        vgg_layer3_out_tensor_name = 'layer3_out:0'
        vgg_layer4_out_tensor_name = 'layer4_out:0'
        vgg_layer7_out_tensor_name = 'layer7_out:0'

        tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

        image_input = tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)
        keep_prob = tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name)
        layer3 = tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)
        layer4 = tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)
        layer7 = tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)

        return image_input, keep_prob, layer3, layer4, layer7

    def create_decoder(self, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out):
        # 1x1 convolution of vgg layer 7
        layer7a_out = tf.layers.conv2d(vgg_layer7_out, self.no_of_classes, 1,
                                       padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        # upsample
        layer4a_in1 = tf.layers.conv2d_transpose(layer7a_out, self.no_of_classes, 4,
                                                 strides=(2, 2),
                                                 padding='same',
                                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        # make sure the shapes are the same!
        # 1x1 convolution of vgg layer 4
        layer4a_in2 = tf.layers.conv2d(vgg_layer4_out, self.no_of_classes, 1,
                                       padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        # skip connection (element-wise addition)
        layer4a_out = tf.add(layer4a_in1, layer4a_in2)
        # upsample
        layer3a_in1 = tf.layers.conv2d_transpose(layer4a_out, self.no_of_classes, 4,
                                                 strides=(2, 2),
                                                 padding='same',
                                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        # 1x1 convolution of vgg layer 3
        layer3a_in2 = tf.layers.conv2d(vgg_layer3_out, self.no_of_classes, 1,
                                       padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        # skip connection (element-wise addition)
        layer3a_out = tf.add(layer3a_in1, layer3a_in2)
        # upsample
        last_layer = tf.layers.conv2d_transpose(layer3a_out, self.no_of_classes, 16,
                                                strides=(8, 8),
                                                padding='same',
                                                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        return last_layer

