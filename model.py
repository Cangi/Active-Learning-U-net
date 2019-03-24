import tensorflow as tf
import math
    
class Model():
    
    def __init__(self,name):

        self.X = tf.placeholder(tf.float32, [None, 128, 128, 3])
        self.Y_ = tf.placeholder(tf.float32, [None, 128, 128, 1])
        self.lr = tf.placeholder(tf.float32)
        self.conv0 = self.__conv2d(self.X, 32, 1, "Y0" + name)#128
        self.conv1 = self.__conv2d(self.conv0, 64, 3, "Y1" + name, strides=(2, 2))#64
        self.conv2 = self.__conv2d(self.conv1, 128, 3, "Y2" + name, strides=(2, 2))#32
       
        self.embedding_size = self.conv2.shape[0]

        self.deconv0 = self.__deconv2d(self.conv2,1,32,128,128,'Y2_deconv' + name)#32
        self.relu0 = tf.nn.relu(self.deconv0)

        self.deconv1 = self.__deconv2d(self.relu0, 2, 64, 64, 128, "Y1_deconv" + name, strides=[1, 2, 2, 1])#64
        self.relu1 = tf.nn.relu(self.deconv1)

        self.deconv2 = self.__deconv2d(self.relu1, 2, 128, 32, 64, "Y0_deconv" + name, strides=[1, 2, 2, 1])#128
        self.relu2 = tf.nn.relu(self.deconv2)

        self.logits = self.__deconv2d(self.relu2, 1, 128, 1, 32, "logits_deconv" + name)#128
        self.loss = tf.losses.sigmoid_cross_entropy(self.Y_, self.logits)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def __deconv2d(self, input_tensor, filter_size, output_size, out_channels, in_channels, name, strides = [1, 1, 1, 1]):
        dyn_input_shape = tf.shape(input_tensor)
        batch_size = dyn_input_shape[0]
        out_shape = tf.stack([batch_size, output_size, output_size, out_channels])
        filter_shape = [filter_size, filter_size, out_channels, in_channels]
        w = tf.get_variable(name=name, shape=filter_shape)
        h1 = tf.nn.conv2d_transpose(input_tensor, w, out_shape, strides, padding='SAME')
        return h1
    def __conv2d(self, input_tensor, depth, kernel, name, strides=(1, 1), padding="SAME"):
        return tf.layers.conv2d(input_tensor, filters=depth, kernel_size=kernel, strides=strides, padding=padding, activation=tf.nn.relu, name=name)

    def sigmoid(self, x):
        if x < 0:
            return 1 - 1/(1 + math.exp(x))
        else:
            return 1 / (1 + math.exp(-x))
        #return 1 / (1 + math.exp(-x))
