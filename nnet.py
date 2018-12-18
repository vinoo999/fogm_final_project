import tensorflow as tf
import numpy as np
import time
import copy
import scf.clustering
tfd = tf.contrib.distributions

class NNetWrapper():
    def __init__(self, data=None, num_clusters=10, cluster_type='soft'):
        if data is None:
            raise ValueError("Data cannot be null")
        elif not isinstance(data, np.ndarray):
            raise ValueError("Data must be of type np.ndarray")

        self.data = data
        self.n_cells = data.shape[0]
        self.n_genes = data.shape[1]
        self.num_clusters = num_clusters
        self.embeddings = None
        self.assignments = None
        self.nnet = AutoEncoderNNet(self.n_genes, cluster_type=cluster_type, batch_size=self.n_cells)
    
    def train(self):
        self.nnet.train(self.data)
        self.embeddings, self.assignments = self.nnet.predict(self.data, get_embeddings=True, get_assignments=True)
        self.nnet.close()

    def get_embeddings(self):
        if self.embeddings is None:
            raise ValueError("Model not trained")

        return self.embeddings

    def get_clusters(self):

        kmeans_labels = clustering.get_kmeans_labels(self.embeddings, self.num_clusters)
        return kmeans_labels


        # if self.assignments is None:
        #     raise ValueError("Model not trained")

        # return np.argmax(self.assignments, axis=1)


class AutoEncoderNNet():
   
    def __init__(self, ftr_size, load_file=None, cluster_type='soft', hyperparams=None, save_loc='/tmp/logs/pcmf/', batch_size=None):

        if hyperparams is None:
                hyperparams = { 'num_epochs': 50,
                                'batch_size': 200,
                                'num_classes': 10,
                                'drop_prob': 0.5,
                                'optimizer': 'adam',
                                'alpha': 0.01,
                                'beta1': 0.9,
                                'beta2': 0.99,
                                'K': 10
                            }



        self.save_loc = save_loc
        self.hyperparams = hyperparams
        if batch_size is not None:
            self.hyperparams['batch_size'] = batch_size
        self.cluster_type = cluster_type
        # Set-up functions necessary
        Relu = tf.nn.relu
        Tanh = tf.nn.tanh
        BatchNormalization = tf.layers.batch_normalization
        Dropout = tf.layers.dropout
        Dense = tf.layers.dense
        Softmax = tf.nn.softmax

        # Neural Net
        self.graph = tf.Graph()
        self.saver = None
        self.sess = None

        # Because I copied and pasted so much
        alpha = self.hyperparams['alpha']
        beta1 = self.hyperparams['beta1']
        beta2 = self.hyperparams['beta2']
        batch_size = self.hyperparams['batch_size']

        if load_file is not None:
            self.load(load_file)
            return

        with self.graph.as_default(): 
            with tf.name_scope('input'):
                self.cells = tf.placeholder(dtype=tf.float32, shape=[batch_size, ftr_size], name="cells")
                self.is_train = tf.placeholder(dtype=tf.bool, shape=(), name="is_train")

            with tf.name_scope('input_layer'):
                input_layer = tf.reshape(self.cells, [-1, ftr_size])
            
            with tf.name_scope('dense'):
                fc1 = Relu(BatchNormalization(Dense(input_layer, 1024), axis=1, training=self.is_train), name='dense1024')
                fc2 = Relu(BatchNormalization(Dense(fc1, 512), axis=1, training=self.is_train), name='dense512')
                fc3 = Relu(BatchNormalization(Dense(fc2, 256), axis=1, training=self.is_train), name='dense256')
                fc4 = Relu(BatchNormalization(Dense(fc3, 128), axis=1, training=self.is_train), name='dense128')
            
            with tf.name_scope('outputs'):
                self.embeddings = tf.identity(fc4, name='embeddings')
                self.cluster_assignments = Softmax(Dense(fc4, self.hyperparams['K']), name='cluster_assignments')
                self.clusters = self.calculate_clusters(self.embeddings, self.cluster_assignments, cluster_type = self.cluster_type)
            
            
            with tf.name_scope('generator'):
                fc4 = Relu(BatchNormalization(Dense(self.embeddings, 256), axis=1, training=self.is_train), name='gen256')
                fc5 = Relu(BatchNormalization(Dense(fc4, 512), axis=1, training=self.is_train), name='gen512')
                fc6 = Relu(BatchNormalization(Dense(fc5, 1024), axis=1, training=self.is_train), name='gen1024')
                self.generation = Dense(fc6, ftr_size, name='generation')
                
            
            
            with tf.name_scope('loss'):
                self.loss, self.loss1, self.loss2, self.loss3 = self.calculate_loss(self.embeddings, self.clusters, self.cluster_assignments, self.cells, self.generation, cluster_type=self.cluster_type)
                tf.summary.scalar("loss", self.loss)
                tf.summary.scalar('cosine_distance', self.loss1)
                tf.summary.scalar('our_loss', self.loss2)
                tf.summary.scalar('cluster_distance', self.loss3)
            
            if self.hyperparams['optimizer'] == 'adam':
                optimizer = tf.train.AdamOptimizer(alpha,beta1,beta2)
            elif self.hyperparams['optimizer'] == 'gd':
                optimizer = tf.train.GradientDescentOptimizer(alpha)

            grads = optimizer.compute_gradients(self.loss)
            self.train_op = optimizer.apply_gradients(grads)
            
            #self.predict_op = tf.argmax(self.cluster_assignments, 1, name="predict_op")
            self.summary_op = tf.summary.merge_all()
        


    def train(self, X_train):
        self.sess = tf.Session(graph=self.graph)
        with tf.Session() as temp_sess:
            temp_sess.run(tf.global_variables_initializer())
        self.sess.run(tf.variables_initializer(self.graph.get_collection('variables')))
    
        summary_writer = tf.summary.FileWriter(self.save_loc, self.sess.graph)
        
        step=0
        batch_size = self.hyperparams['batch_size']
        start_time = time.time()
        for epoch in range(self.hyperparams['num_epochs']):

            batch_num = 0
            # print("epoch:", epoch)
            for i in range(X_train.shape[0]//batch_size + 1):
                X_batch, batch_num = self.get_batch(X_train, batch_size, batch_num)
                if len(X_batch) != batch_size:
                    continue
                input_dict = {self.cells: X_batch, self.is_train: True}
                _, loss_value, loss1_val, loss2_val, loss3_val, summary_str = self.sess.run([self.train_op, self.loss, self.loss1, self.loss2, self.loss3, self.summary_op], feed_dict=input_dict)

                # Write the summaries and print an overview fairly often.
                if step % 20 == 0:
                    duration = time.time()-start_time
                    print('Step {:.0f}: loss = {:.3f},  ({:.3f} sec)'.format(step, loss_value, duration))
                    # print(loss1_val)
                    # print(loss2_val)
                    # print(loss3_val)

                    # Update the events file.
                    summary_writer.add_summary(summary_str, global_step=step)
                    summary = tf.Summary()
                    summary_writer.add_summary(summary, global_step=step)

                # Save a checkpoint periodically.
                if (step+1) % 100 == 0:
                    print('Saving')
                    fname = self.save_loc + 'checkpoint' + str(step) + '.pth.tar'
                    self.save(fname)

                step += 1

        print('Saving')
        fname = self.save_loc + 'checkpoint' + str(step) + '.pth.tar'
        self.save(fname)
        print('Done training for {:.0f} epochs, {:.0f} steps.'.format(self.hyperparams['num_epochs'], step))

        # self.sess.close()


    def predict(self, X_val, get_embeddings=False, get_assignments=False, get_centroids=False):
        val_batch_num = 0
        all_embeddings = None
        all_assignments = None
        all_clusters = None
        batch_size = X_val.shape[0] # self.hyperparams['batch_size']
        for j in range(len(X_val) // batch_size):
            X_batch_val, val_batch_num = self.get_batch(X_val, batch_size, val_batch_num)
            if len(X_batch_val) != batch_size:
                continue

            ops = []
            if get_embeddings:
                ops.append(self.embeddings)
            if get_assignments:
                ops.append(self.cluster_assignments)
            if get_centroids:
                ops.append(self.clusters)

            batch_output = self.sess.run(ops, feed_dict={self.cells: X_batch_val, self.is_train: False})
            
            index = 0
            if get_embeddings:
                batch_embeddings = batch_output[index]
                if all_embeddings is None:
                    all_embeddings = batch_embeddings
                else:
                    all_embeddings = np.concatenate((all_embeddings, batch_embeddings),0)
                index+=1
            if get_assignments:
                batch_assignments = batch_output[index]
                if all_assignments is None:
                    all_assignments = batch_assignments
                else:
                    all_assignments = np.concatenate((all_assignments, batch_assignments),0)
                index+=1
            if get_centroids:
                batch_clusters = batch_output[index]
                if all_clusters is None:
                    all_clusters = batch_clusters
                else:
                    all_clusters = batch_clusters # fix later

        outputs = []
        if get_embeddings:
            outputs.append(all_embeddings)
        if get_assignments:
            outputs.append(all_assignments)
        if get_centroids:
            outputs.append(all_clusters)

        return outputs



    def save(self, fname='checkpoint.tf'):
        if self.saver is None:
            self.saver = tf.train.Saver(self.graph.get_collection('variables'))
        with self.graph.as_default():
            self.saver.save(self.sess, fname)

    def load(self, fname):
        with self.graph.as_default():
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, fname)


    def close(self):
        self.sess.close()


    def calculate_loss(self,embeddings, clusters, soft_assignments, X, Gen, cluster_type='soft'):
        '''
        Embeddings: NxD
        Clusters: KxD
        Soft Assignments: NxK
        Return loss
        '''
        
        N = embeddings.shape[0]
        D = embeddings.shape[1]
        K = clusters.shape[0]

        KND_Mat = tf.tile(tf.reshape(tf.expand_dims(clusters, axis=0), (K,-1,D)), multiples=(1,N,1))
        
        loss1 = tf.multiply(tf.losses.cosine_distance(X, Gen, axis=0, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS), 1.0/(int(N)*int(D)*(int(K))))
        
        distances = tf.reduce_sum(tf.square(embeddings - KND_Mat), axis=2)
        if cluster_type == 'soft':
            loss2 = tf.multiply(tf.reduce_sum(tf.multiply(soft_assignments, tf.transpose(distances))), 1.0/(int(N)*int(K)*int(D)))
        elif cluster_type == 'hard':
            loss2 = tf.multiply(tf.reduce_sum(tf.multiply(tf.one_hot(tf.argmax(soft_assignments, axis=1), depth = K), tf.transpose(distances))), 1.0/(int(N)*int(K)*int(D)))
        
        lhs = tf.expand_dims(clusters, axis=1)
        rhs = tf.expand_dims(clusters, axis=2)
        loss3 = tf.multiply(tf.reduce_sum(tf.norm(tensor=(lhs - rhs), ord=1, axis=(-2, -1))), -1.0/(int(K)*int(D)))
        


        loss = loss1 + loss2 + loss3
        return loss, loss1, loss2, loss3
        
    
    def calculate_clusters(self, embeddings, soft_assignments, cluster_type='soft'):
        '''
        Embeddings: NxD
        Soft Assignments NxK
        Return KxD vector of clusters
        '''
        K = soft_assignments.shape[1]
        D = embeddings.shape[1]
        eps = 1e-3
        if cluster_type == 'soft':
            counts = tf.transpose(tf.reduce_sum(soft_assignments, axis=0))
            augmented_counts = tf.expand_dims(counts, axis=1)
            augmented_counts = tf.tile(augmented_counts, multiples=(1,D))
    #         augmented_counts = tf.tile(counts, multiples(K,1))
            return tf.matmul(tf.transpose(soft_assignments),embeddings, name='clusters')/(augmented_counts+eps)
        if cluster_type == 'hard':
            counts = tf.transpose(tf.reduce_sum(tf.one_hot(tf.argmax(soft_assignments, axis=1), depth=K),axis=0))
            augmented_counts = tf.expand_dims(counts, axis=1)
            augmented_counts = tf.tile(augmented_counts, multiples=(1,D))
            return tf.matmul(tf.transpose(tf.one_hot(tf.argmax(soft_assignments, axis=1), depth=K)), embeddings, name='clusters')/(augmented_counts+eps)

        
    def get_batch(self, X, batch_size, batch_num):
        """
        Return minibatch of samples and labels

        :param X, y: samples and corresponding labels
        :parma batch_size: minibatch size
        :returns: X_batch
        """
        new_start = batch_size * batch_num
        if new_start >= X.shape[0]:
            new_start = 0
            batch_num = 0
        new_end = new_start + batch_size
        batch_num += 1
        if new_end >= X.shape[0]:
            new_end = X.shape[0]
            batch_num = 0
        X_batch = X[np.arange(new_start, new_end), :]
        return X_batch, batch_num



