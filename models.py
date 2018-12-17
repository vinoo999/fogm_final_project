import numpy as np
import edward as ed
import tensorflow as tf 
from edward.models import Gamma, Poisson, PointMass, Uniform, Dirichlet, Empirical, Categorical, Mixture, Independent
import scf.clustering

tfd = tf.contrib.distributions

class PoissonMF(object):
    def __init__(self, data = None, latent_dim = 10, num_clusters=2):
        self.graph = tf.get_default_graph()
        self.sess = tf.Session(graph=self.graph)
        if data is None:
            raise ValueError("Data cannot be null")
        elif not isinstance(data, np.ndarray):
            raise ValueError("Data must be of type np.ndarray")

        self._trained = False 
        self.data = data
        self.num_clusters = num_clusters
        self.latent_dim = latent_dim
        self.n_cells = data.shape[0]
        self.n_genes = data.shape[1]

        print("n_cells:", self.n_cells)
        print("n_genes:", self.n_genes)

        self.U_f = None
        self.V_f = None
    
    def train(self, n_iter=1000):
        with self.sess:
            self.alpha = np.ones(self.num_clusters, dtype=np.float32)
            self.pi = Dirichlet(self.alpha)
            self.z = Categorical(probs=self.pi, sample_shape=self.n_cells)
            
            self.rV = Gamma(0.1, 0.1, sample_shape=self.n_genes)

            self.rU = Gamma(tf.ones(self.latent_dim), tf.ones(self.latent_dim), sample_shape=self.num_clusters)
            
            # components = [Independent(distribution=Gamma(tf.ones(self.latent_dim), rU[k], sample_shape=self.n_cells), reinterpreted_batch_ndims=1) for k in range(self.num_clusters)]
            # print("comp: ", components[0].shape, len(components))
            # U = tf.convert_to_tensor([components[ tf.Variable(z[i]) ] for i in range(self.n_cells)])
            self.rUstar = tf.transpose(tf.gather(self.rU, self.z))
            self.U = Gamma(concentration=tf.ones(self.rUstar.shape), rate=self.rUstar)
            print("U: ",self.U.shape)
            print("z:", self.z.shape)
            
            # U = Gamma(concentration=1.0, rate=rU, sample_shape=self.latent_dim)
            self.V = Gamma(concentration=1.0, rate=self.rV, sample_shape=self.latent_dim)
            self.X = Poisson(tf.matmul(tf.transpose(self.U), self.V))

            # INFERENCE
            self.qpi = PointMass(tf.get_variable(
                    "qpi/params", [self.num_clusters],
                    initializer=tf.constant_initializer(1.0 / self.num_clusters)))
            self.qz = PointMass(tf.get_variable(
                "qz/params", [self.n_cells],
                initializer=tf.zeros_initializer(),
                dtype=tf.int32))
            self.qrU = PointMass(tf.nn.softplus(tf.Variable(tf.random_normal([self.num_clusters, self.latent_dim]))),)
            self.qrV = PointMass(tf.nn.softplus(tf.Variable(tf.random_normal([self.n_genes]))),)

            self.qU = PointMass(
                tf.nn.softplus(tf.Variable(tf.random_normal([self.latent_dim, self.n_cells]))),
            )
            
            self.qV = PointMass(
                tf.nn.softplus(tf.Variable(tf.random_normal([self.latent_dim, self.n_genes]))),
            )

            print("X", self.X.shape)
            print("data", self.data.shape, type(self.data))

            # print([n.name for n in tf.get_default_graph().as_graph_def().node])
    
            # inference_e = ed.HMC(
            #     {self.pi:self.qpi, self.z:self.qz, self.rU:self.qrU, self.rV:self.qrV}, 
            #     data={self.X: self.data, self.U:self.qU, self.V:self.qV},
            # )

            # inference_m = ed.MAP(
            #     {self.U:self.qU, self.V:self.qV},
            #     data={self.X: self.data, self.pi:self.qpi, self.z:self.qz, self.rU:self.qrU, self.rV:self.qrV},
            # )

            # inference_e.initialize(n_iter=n_iter, optimizer="rmsprop")
            # inference_m.initialize(n_iter=n_iter, optimizer="rmsprop")
            

            inference = ed.MAP({self.U: self.qU, self.V: self.qV, self.pi:self.qpi, self.z:self.qz, self.rU:self.qrU, self.rV:self.qrV}, data={self.X: self.data})
            inference.initialize(n_iter=n_iter, optimizer=tf.train.AdamOptimizer
                             (learning_rate=0.001, beta1=0.9, beta2=0.999,
                              epsilon=1e-08))
            tf.global_variables_initializer().run()
            self.loss = np.empty(n_iter, dtype=np.float32)
            for i in range(n_iter):
                # info_dict_e = inference_e.update()
                # info_dict_m = inference_m.update()
                info_dict = inference.update()
                # loss[i] = info_dict_m["loss"]
                self.loss[i] = info_dict["loss"]
                inference.print_progress(info_dict)
            self.info_dict = info_dict
            self._trained = True
            # sess = ed.get_session()
            self.U_f = self.sess.run(self.qU)
            self.V_f = self.sess.run(self.qV)
    
    def get_embeddings(self):
        if self._trained:
            return self.U_f
        else:
            raise ValueError("Model not trained")

class PoissonFactor(object):
    def __init__(self, data = None, latent_dim = 10, num_clusters=2):
        self.graph = tf.get_default_graph()
        self.sess = tf.Session(graph=self.graph)
        if data is None:
            raise ValueError("Data cannot be null")
        elif not isinstance(data, np.ndarray):
            raise ValueError("Data must be of type np.ndarray")

        self._trained = False 
        self.data = data
        self.num_clusters = num_clusters
        self.latent_dim = latent_dim
        self.n_cells = data.shape[0]
        self.n_genes = data.shape[1]
        self.U = None
        self.V = None

    def train(self, n_iter=1000):
        with self.sess:
            U = Gamma(concentration=1.0, rate=1.0, sample_shape=[self.n_cells, self.latent_dim])
            V = Gamma(concentration=1.0, rate=1.0, sample_shape=[self.n_genes, self.latent_dim])
            R = Poisson(tf.matmul(U, V, transpose_b=True))

            qU = PointMass(tf.nn.softplus(tf.Variable(tf.random_normal([self.n_cells, self.latent_dim]))))
            qV = PointMass(tf.nn.softplus(tf.Variable(tf.random_normal([self.n_genes, self.latent_dim]))))

            inference = ed.MAP({U: qU, V: qV}, data={R: self.data})
            inference.initialize(optimizer=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999,
                epsilon=1e-08))
            tf.global_variables_initializer().run()

            loss = np.empty(n_iter, dtype=np.float32)
            for i in range(n_iter):
                info_dict = inference.update()
                loss[i] = info_dict["loss"]
                inference.print_progress(info_dict)

            self._trained = True

            # sess = ed.get_session()

            self.U = self.sess.run(qU)
            self.V = self.sess.run(qV)

    def get_embeddings(self):
        if self._trained:
            return self.U
        else:
            raise ValueError("Model not trained")

    def get_clusters(self):
        if self._trained:
            kmeans_labels = clustering.get_kmeans_labels(self.U, self.num_clusters)
            return kmeans_labels
        else:
            raise ValueError("Model not trained")


class PoissonHierarchical(object):
    def __init__(self, data=None, latent_dim = 10, num_clusters=2):
        self.graph = tf.get_default_graph()
        self.sess = tf.Session(graph=self.graph)
        if data is None:
            raise ValueError("Data cannot be null")
        elif not isinstance(data, np.ndarray):
            raise ValueError("Data must be of type np.ndarray")

        self._trained = False 
        self.data = data
        self.latent_dim = latent_dim
        self.n_cells = data.shape[0]
        self.n_genes = data.shape[1]
        self.U = None
        self.V = None
        self.num_clusters = num_clusters


    def train(self, n_iter = 500, t = 500):
        with self.sess:
            cFeat = Gamma(1.0, 1.0, sample_shape=self.n_cells) # Users cFeativity
            U = Gamma(1.0, cFeat, sample_shape=self.latent_dim) # Users Uerence

            gFeat = Gamma(0.1, 0.1, sample_shape=self.n_genes) # Items gFeatularity
            V = Gamma(1.0, gFeat, sample_shape=self.latent_dim) # Items Vibute

            R = Poisson(tf.matmul(U, V, transpose_a=True))

            qcFeat = Empirical(
                tf.nn.softplus(tf.Variable(tf.random_normal([t, self.n_cells]))),
            )
            qU = PointMass(
                tf.nn.softplus(tf.Variable(tf.random_normal([self.latent_dim, self.n_cells]))),
            )
            qgFeat = Empirical(
                tf.nn.softplus(tf.Variable(tf.random_normal([t, self.n_genes]))),
            )
            qV = PointMass(
                tf.nn.softplus(tf.Variable(tf.random_normal([self.latent_dim, self.n_genes]))),
            )

            inference_e = ed.Gibbs(
                {cFeat: qcFeat, gFeat: qgFeat}, 
                data = {R: self.data, U: qU, V: qV},
            )

            inference_m = ed.MAP(
                {U: qU, V: qV},
                data = {R: self.data, cFeat: qcFeat, gFeat: qgFeat},
            )

            inference_e.initialize()
            inference_m.initialize(n_iter=n_iter, optimizer="rmsprop")

            tf.global_variables_initializer().run()


            loss = np.empty(n_iter, dtype=np.float32)

            for i in range(n_iter):
                info_dict_e = inference_e.update()
                info_dict_m = inference_m.update()
                
                loss[i] = info_dict_m["loss"]
                
                inference_m.print_progress(info_dict_m)

            self._trained = True
            # sess = ed.get_session()
            self.U = self.sess.run(qU)
            self.V = self.sess.run(qV)

    def get_embeddings(self):
        if self._trained:
            return self.U
        else:
            raise ValueError("Model not trained")

    def get_clusters(self):
        if self._trained:
            kmeans_labels = clustering.get_kmeans_labels(self.U, self.num_clusters)
            return kmeans_labels
        else:
            raise ValueError("Model not trained")

class PoissonMixture(object):
    def __init__(self, data = None, num_clusters = 2, latent_dim = 10):
        self.graph = tf.get_default_graph()
        self.sess = tf.Session(graph=self.graph)
        if data is None:
            raise ValueError("Data cannot be null")
        elif not isinstance(data, np.ndarray):
            raise ValueError("Data must be of type np.ndarray")

        self._trained = False 
        self.data = data
        self.latent_dim = latent_dim
        self.n_cells = data.shape[0]
        self.n_genes = data.shape[1]
        self.U = None
        self.V = None
        self.num_clusters = num_clusters
        self.qcat_est = None

    def train(self, n_iter=1000):
        with self.sess:
            alpha = np.random.uniform(low=0.4, high=10, size = self.num_clusters).astype(np.float32)

            pi = Dirichlet(alpha)
            cat = Categorical(logits=tf.log(pi), sample_shape=self.n_cells)
            U = Gamma(concentration=1.0, rate=1.0,
                    sample_shape=[self.num_clusters, self.n_cells, self.latent_dim])
            V = Gamma(concentration=1.0, rate=1.0,
                    sample_shape=[self.n_genes, self.latent_dim])

            V_reps = tf.transpose(tf.tile(
                tf.expand_dims(V, axis=0), [self.n_cells, 1, 1]), [0, 2, 1])
            U_all = tf.gather(U, cat)

            idxs = tf.constant(np.repeat(
                np.arange(self.n_cells), 2).reshape(self.n_cells, -1))
            res = tf.matmul(U_all, V_reps)
            res_sel = tf.gather_nd(res, idxs)

            R = Poisson(res_sel)

            qU = PointMass(tf.nn.softplus(tf.Variable(
                tf.random_normal([self.num_clusters, self.n_cells, self.latent_dim]))))
            qV = PointMass(tf.nn.softplus(tf.Variable(
                tf.random_normal([self.n_genes, self.latent_dim]))))

            # sess = ed.get_session()
            self.sess.run(tf.global_variables_initializer())
            qu_bt = self.sess.run(qU)
            qv_bt = self.sess.run(qV)
            qpi_est = self.sess.run(pi)
            qcat_est = self.sess.run(cat)

            inference_m = ed.MAP({U: qU, V: qV}, data={R: self.data, pi: qpi_est})

            inference_m.initialize(optimizer="rmsprop")
            self.sess.run(tf.global_variables_initializer())

            loss = np.empty(n_iter, dtype=np.float32)
            for i in range(n_iter):
                info_dict_m = inference_m.update()
                loss[i] = info_dict_m["loss"]
                inference_m.print_progress(info_dict_m)

            # sess = ed.get_session()
            U_est = self.sess.run(qU)
            self.V = self.sess.run(qV)
            self.qcat_est = self.sess.run(cat)
            U_final = []
            for idx, choice in enumerate(self.qcat_est):
                U_final.append(U_est[choice][idx])
            self.U = np.array(U_final)

            self._trained = True

    def get_embeddings(self):
        if self._trained:
            return self.U
        else:
            raise ValueError("Model not trained")

    def get_clusters(self):
        if self._trained:
            kmeans_labels = clustering.get_kmeans_labels(self.U, self.num_clusters)
            return kmeans_labels
        else:
            raise ValueError("Model not trained")
