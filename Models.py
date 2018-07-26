from keras.models import Model, load_model, model_from_json
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Reshape, UpSampling2D, concatenate
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU
import os

class MLModel (object):
    def __init__(self,
                 train,
                 test,
                 parametri,
                 train_again=True):
        self.kind = parametri
        self.n_classes = test.n_cl
        self.classifier = None
        self.x_tr, self.y_tr, self.x_te, self.y_te = train.x, train.y, test.x, test.y
        self.weithts = train.weights
        self._model_path = 'models_backup/' +parametri+'.hdf5'
        self._log_path = 'logs/' +parametri
        self.train_again = train_again

    def init(self,
             n_filters = 16,
             n_dense = 256,
             learning_rate=0.001,
             loss_type='mean_squared_error'):
        from io import StringIO
        import sys
        # CNN Model
        self.n_filters = n_filters
        self.lr = learning_rate
        self.n_dense = n_dense
        self.loss_type = loss_type
        if 'CNN' in self.kind:
            self.classifier = self.get_CNN()
        if 'FFNN' in self.kind:
            self.classifier = self.get_FFNN ()
        # save formatted summary from standard output
        old_stdout = sys.stdout
        result = StringIO ()
        sys.stdout = result
        self.classifier.summary ()
        sys.stdout = old_stdout
        self._summ = result.getvalue ()
        return

    @property
    def summ(self):
        return self._summ

    @property
    def log_path(self):
        return self._log_path

    @property
    def model_path(self):
        return self._model_path

    def train_classifier(self, epochs=50, batch_size=128):
        if 'FFNN' in self.kind:
            kind = self.kind
            tensorboard = TensorBoard (log_dir=self._log_path,
                                       histogram_freq=0,
                                       write_graph=True,
                                       write_images=True)
            if not os.path.exists ('models_backup'):
                os.makedirs ('models_backup')
            checkpoint = ModelCheckpoint (self._model_path,
                                          monitor='val_acc',
                                          verbose=1,
                                          save_best_only=True,
                                          mode='max')
            if self.train_again:
                hist = self.classifier.fit (self.x_tr, self.y_tr,
                                            epochs=epochs,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            validation_data=(self.x_te, self.y_te),
                                            class_weight=self.weithts,
                                            callbacks=[tensorboard, checkpoint])
            self.load_best_model()
            return hist

        if 'CNN' in self.kind:
            kind = self.kind
            tensorboard = TensorBoard(log_dir=self._log_path,
                                      histogram_freq=0,
                                      write_graph=True,
                                      write_images=True)
            if not os.path.exists('models_backup'):
                os.makedirs('models_backup')
            checkpoint = ModelCheckpoint(self._model_path,
                                         monitor='val_acc',
                                         verbose=1,
                                         save_best_only=True,
                                         mode='max')

            hist = self.classifier.fit (self.x_tr, self.y_tr,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        validation_data=(self.x_te, self.y_te),
                                        class_weight=self.weithts,
                                        callbacks=[tensorboard, checkpoint])
            self.load_best_model()
            return hist

    def load_best_model(self):
        # FFNN
        if 'FFNN' in self.kind:
            model = self.get_FFNN ()
        # CNN
        if 'CNN' in self.kind:
            model = self.get_CNN ()
        model.load_weights (self._model_path)
        self.classifier = model

    def get_FFNN(self):
        n_filters = self.n_filters
        learning_rate = self.lr
        n_dense = self.n_dense
        input_shape = [self.x_tr[0].shape[0], self.x_tr[0].shape[1], 1]
        input_img = Input (shape=input_shape)

        x = Flatten (input_shape=input_shape) (input_img)
        x = Dense (n_dense, activation='relu') (x)
        x = Dense (n_dense, activation='relu') (x)
        x = Dropout (0.2) (x)
        x = Dense (self.n_classes, activation='softmax') (x)

        model = Model(input_img, x)
        opt = Adam (lr=learning_rate)
        # opt = SGD (lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
        model.compile (optimizer=opt, loss=self.loss_type, metrics=['accuracy'])

        return model

    def get_CNN(self):
        n_filters = self.n_filters
        learning_rate = self.lr
        n_dense = self.n_dense
        input_shape = [self.x_tr[0].shape[0], self.x_tr[0].shape[1], 1]
        input_img = Input (shape=input_shape)
        x = Conv2D (
            kernel_size = (3,3),
            # strides=(2,2),
            padding='same',
            filters = n_filters,
            activation='relu',
            input_shape=input_shape) (input_img)
        x = Conv2D (
            kernel_size = (2,2),
            padding='same',
            filters=n_filters,
            activation='relu') (x)
        x = Conv2D (
            kernel_size = (2,2),
            padding='same',
            filters=n_filters,
            activation='relu') (x)

        x = Dropout (0.3) (x)
        # x = Conv2D (filters=n_filters*2, kernel_size=(3, 3), padding='same', activation='relu') (x)
        # x = Dropout (0.3) (x)

        x = Flatten () (x)
        # x = Dense (n_dense, activation='relu') (x)
        # x = Dropout (0.2) (x)
        x = Dense (self.n_classes, activation='softmax') (x)

        model = Model(input_img, x)
        opt = Adam (lr=learning_rate)
        # opt = SGD (lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
        model.compile (optimizer=opt, loss=self.loss_type, metrics=['accuracy'])

        return model
