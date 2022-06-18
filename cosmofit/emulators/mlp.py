import numpy as np
import scipy as sp

from .base import BaseEmulatorEngine


class MLPEmulatorEngine(BaseEmulatorEngine):

    def __init__(self, pipeline, validation_frac=0.2, nhidden=(100, 100, 100), optimizer='adam', batch_sizes=(320, 640, 1280, 2560, 5120), epochs=100, learning_rates=(1e-2, 1e-3, 1e-4, 1e-5, 1e-6), ytransform='arcsinh', npcs=None, seed=None, **kwargs):
        super(MLPEmulatorEngine, self).__init__(pipeline=pipeline, **kwargs)
        self.validation_frac = float(validation_frac)
        self.rng = np.random.RandomState(seed=seed)
        self.nhidden = tuple(nhidden)
        self.optimizer = str(optimizer)
        self.batch_sizes = tuple(batch_sizes)
        self.learning_rates = tuple(learning_rates)
        self.epochs = epochs
        self.npcs = npcs
        self.ytransform = str(ytransform)

    def get_default_samples(self, engine='rqrs', niterations=300):
        from cosmofit.samplers import QMCSampler
        sampler = QMCSampler(self.pipeline, engine=engine)
        sampler.run(niterations=niterations)
        return sampler.samples

    def fit(self):

        self.operations, self.yshapes = None, None

        import tensorflow as tf
        class Model(tf.keras.Model):

            def __init__(self, architecture, eigenvectors=None, mean=None, sigma=None):
                super(Emulator, self).__init__()
                self.architecture = architecture
                self.nlayers = len(self.architecture) - 1
                self.mean = mean
                self.sigma = sigma
                self.eigenvectors = eigenvectors

                self.W, self.b, self.alpha, self.beta = [], [], [], []
                for i in range(self.nlayers):
                    self.W.append(tf.Variable(tf.random.normal([self.architecture[i], self.architecture[i + 1]], 0., np.sqrt(2. / len(self.architecture[0]))), name='W_{:d}'.format(i), trainable=True))
                    self.b.append(tf.Variable(tf.zeros([self.architecture[i+1]]), name='b_{:d}'.format(i), trainable=True))
                for i in range(self.nlayers-1):
                    self.alpha.append(tf.Variable(tf.random.normal([self.architecture[i + 1]]), name='alpha_{:d}'.format(i), trainable=True))
                    self.beta.append(tf.Variable(tf.random.normal([self.architecture[i + 1]]), name='beta_{:d}'.format(i), trainable=True))

            @tf.function
            def call(self, params):
                x = params
                for i in range(self.nlayers):
                    # linear network operation
                    x = tf.add(tf.matmul(x, self.W[i]), self.b[i])
                    # non-linear activation function
                    if i < self.nlayers - 1:
                        x = tf.multiply(tf.add(self.beta[i], tf.multiply(tf.sigmoid(tf.multiply(self.alpha[i], x)), tf.subtract(1., self.beta[i]))), x)
                # linear output layer
                if self.eigenvectors is not None:
                    x = tf.matmul(tf.add(tf.multiply(x, self.sigma), self.mean), self.eigenvectors.T)
                return x

            def operations(self):
                operations = []
                for i in range(self.nlayers):
                    # linear network operation
                    operations.append({'eval': 'x @ W + b', 'locals': {'W': self.W[i], 'b': self.b[i]}})
                    # non-linear activation function
                    if i < self.nlayers - 1:
                        operations.append({'eval': '(beta + (sp.special.expit(alpha * x) * (1 - beta))) * x', 'locals': {'alpha': self.alpha[i], 'beta': self.beta[i]}})
                # linear output layer
                if self.eigenvectors is not None:
                    operations.appemd({'eval': 'eigenvectors @ (value * sigma + mean)', 'locals': {'eigenvectors': eigenvectors, 'mean': mean, 'sigma': sigma}})
                return operations


        nsamples = self.mpicomm.bcast(len(self.samples) if self.mpicomm.rank == 0 else None)
        nvalidation = int(nsamples * self.validation_frac + 0.5)
        if nvalidation >= nsamples:
            raise ValueError('Cannot use {:d} validation samples (>= {:d} total samples)'.format(nvalidation, nsamples))

        if self.mpicomm.rank == 0:
            mask = np.zeros(nsamples, dtype='?')
            mask[self.rng.choice(nsamples, size=nvalidation, replace=False, shuffle=False)] = True
            samples = {'X': self.samples.to_array(params=self.samples.params(output=False), struct=False)}
            samples['X'].shape = (samples['X'].shape[0], -1)
            Y, self.yshapes = [], {}
            for name in self.samples.names(output=True):
                self.yshapes[name] = self.samples[name].shape[1:]
                Y.append(self.samples[name].ravel())
            samples['Y'] = np.concatenate(Y, axis=-1)
            self.operations = {}
            for name, value in samples.items():
                mean, sigma = np.mean(value, axis=0), np.std(value, ddof=1, axis=0)
                self.operations[name] = [{'eval': 'x * sigma + mean' if name == 'Y' else '(x - mean} / sigma',
                                         'locals': {'mean': mean, 'sigma': sigma}}]
                setattr(self, name, (value - mean) / sigma)
            if 'arcsinh' in self.ytransform:
                Y = np.arcsinh(samples['Y'])
                mean, sigma = np.mean(Y, axis=0), np.std(Y, ddof=1, axis=0)
                samples['Y'] = (Y - mean) / sigma
                self.operations['Y'].insert(0, {'eval': 'np.sinh(x * sigma + mean)', 'locals': {'mean': mean, 'sigma': sigma}})
            for name, value in samples.items():
                samples['{}_validation'.format(name)] = value[mask]
                samples['{}_training'.format(name)] = value[~mask]
            eigenvectors, mean, sigma = None, None, None
            architecture = [len(self.varied_params)] + list(self.nhidden)
            if self.npcs is not None:
                cov = np.cov(samples['Y_training'], rowvar=True, ddof=1)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                eigenvectors = eigenvectors[:self.npcs]
                tmp = np.dot(samples['Y_training'], eigenvectors)
                mean, sigma = np.mean(tmp, axis=0), np.std(tmp, ddof=1, axis=0)
                architecture += [self.npcs]
            else:
                architecture += [samples['Y'].shape[1]]

            model = Model(architecture, eigenvectors=eigenvectors, mean=mean, sigma=sigma)
            model.compile(optimizer=self.optimizer, loss='mse', metrics=['mse'])
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

            for lr, batch_size in zip(self.learning_rates, self.batch_sizes):
                self.log_info('Using learning rate {:.2e} and batch size {:d}'.format(lr, batch_size))
                model.optimizer.lr = lr
                model.fit(samples['X_training'], samples['Y_training'], epochs=self.epochs, batch_size=nbatch,
                          validation_data=(samples['X_validation'], samples['Y_validation']), callbacks=[es], verbose=2)
                self.operations['M'] = model.operations()
            self.operations = self.operations['X'] + self.operations['M'] + self.operations['Y']

        self.operations = self.mpicomm.bcast(self.operations, root=0)
        self.yshapes = self.mpicomm.bcast(self.yshapes, root=0)

    def predict(self, **params):
        x = [params[param] for param in self.varied_names]
        for operation in self.operations:
            x = eval(transform['eval'], {'np': np, 'sp': sp}, {'x': x, **transform['locals']})
        cumsize, toret = 0, {}
        for name, shape in self.yshapes.items():
            size = np.prod(shape, dtype='i')
            toret[name] = x[cumsize:cumsize + size].reshape(shape)
            cumsize += size
        return toret

    def __getstate__(self):
        state = super(MLPEmulatorEngine, self).__getstate__()
        for name in ['operations', 'yshapes']:
            state[name] = getattr(self, name)
        return state
