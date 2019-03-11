import tensorflow as tf
import numpy as np
import model
import os

class ActiveLearning:

    def __init__(self, train_images, train_labels, labeled_data, model, arguments):
        self.pool_size = len(train_labels)
        self.labeled_data = labeled_data
        self.train_images = train_images
        self.train_labels = train_labels
        self.arguments = arguments
        self.feature_vector = 0
        self.model = model

    def query(self,n):
        pass

    def shuffle(self):
        p = np.random.permutation(len(self.train_images))
        self.train_images = self.train_images[p]
        self.train_labels = self.train_labels[p]
    
    def next_batch(self, batch_s, iters):
        if(iters == 0):
            self.shuffle()
        count = batch_s * iters
        return self.train_images[count:(count + batch_s)], self.train_labels[count:(count + batch_s)]
    
    def reset_model(self):
        pass

    def get_embedings(self, images, labels):
        #Use the same model as the one trainned and reset it after each active loop
        model_for_embedings = model.Model('_embedings')
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        feed_dict = {model_for_embedings.X : images, model_for_embedings.Y_: labels, model_for_embedings.lr: 0.0005}
        reshape = tf.reshape(model_for_embedings.conv2, [-1, (32 * 32 * 128)])
        embedings = sess.run([reshape], feed_dict=feed_dict)
        return embedings

    def update_labeled_data(self,labeled_data):
        self.labeled_data = labeled_data

    def train(self):
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        
        batch_count = 0
        display_count = 1
        for epoch in range(1, 100):

            for i in range(8):
                if(batch_count > 8):
                    batch_count = 0    

                batch_X, batch_Y = self.next_batch(10, batch_count)

                batch_count += 1

                feed_dict = {self.model.X: batch_X, self.model.Y_: batch_Y, self.model.lr: 0.0005}
                loss_value, _ = sess.run([self.model.loss, self.model.optimizer], feed_dict=feed_dict)

                if(i % 500 == 0):
                    print(str(display_count) + " training loss:", str(loss_value))
                    display_count +=1
        save_path = saver.save(sess, os.path.realpath('') + '/trained_models/model_' + str(len(self.train_images)) + '/model.ckpt')
        print("Model saved in path: %s" % save_path)
        print("Done!")

    def predict(self, x, y):
        predicted_label = tf.zeros(len(y), dtype=tf.float64)
        predictions = tf.zeros([len(y), len(np.unique(y))])
        tf.reset_default_graph()
        self.model = model.Model('')
        saver = tf.train.Saver()
        sess = tf.Session()
        path = os.path.realpath('') + '/trained_models/model_' + str(len(self.train_images)) + '/model.ckpt'
        saver.restore(sess, path)

        print('Model loaded from: ' + path)

        ix = 3 #random.randint(0, 64) #len(X_test) - 1 = 64
        test_image = x[ix].astype(float)
        test_image = np.reshape(test_image, [-1, 128 , 128, 3])
        test_data = {self.model.X:test_image,self.model.Y_:y}

        loss,predictions = sess.run([self.model.loss,self.model.logits],feed_dict=test_data)
        predictions = np.reshape(predictions,[-1])
        return loss,predictions, predicted_label