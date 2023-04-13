from sklearn.base import BaseEstimator, RegressorMixin
from keras.callbacks import EarlyStopping
from keras.models import Sequential 
from keras.layers import Dense 

class AE(BaseEstimator, RegressorMixin):
        
    def __init__(
        self,
        n_hidden_layers: int = 1,
        layer_dims: list = [3],
        activation_fs: list = ['relu', 'relu'],
        env_dimensionality: int = 100,
        training_epochs: int = -1,
        loss: str = "mean_squared_error",
        batch_size: int = 1,
        ):
    
        """
        Arguments:
            n_hidden_layers: n of hidden layers
            layer_dims: list containing hidden layers' widths (does not include the sizes of the inputs and outputs)
            activation_fs: list containing activation functions for each layer (must also provide activation
            function for the last, decoding, layer)
            env_dimensionality: dimensionality of the ground truth
            training_epochs: n of training updates (-1 means training until convergence)
            loss: loss function used, see keras options: https://keras.io/api/losses/
            batch_size: batch size for training (cannot be larger than the smallest dataset size)
        """
        self.n_hidden_layers = n_hidden_layers
        self.layer_dims = layer_dims
        self.env_dimensionality = env_dimensionality
        self.training_epochs = training_epochs
        self.loss = loss
        self.batch_size = batch_size
        
        # initialize the AE        
        self.model = Sequential()
        for i in range(len(layer_dims)):
            if i==0:
                self.model.add(Dense(layer_dims[i], input_shape=(self.env_dimensionality,), 
                                     activation = activation_fs[i]))
            else: 
                self.model.add(Dense(layer_dims[i], activation = activation_fs[i]))
        self.model.add(Dense(self.env_dimensionality, activation = activation_fs[i+1])) # output layer
            
        self.model.compile(optimizer='adam', loss=self.loss)
            
            
    def fit(self, X, y=None):
        # y can be different from X (e.g. for a masked autoencoder task)
        if y is None:
            y = X # autoencoding
        
        if self.training_epochs == -1: # train until convergence
            overfitCallback = EarlyStopping(monitor='loss', min_delta=0, patience = 10)

            self.model.fit(X, y,
                epochs=100000000,
                batch_size=self.batch_size,
                shuffle=True,
                validation_data=(X, y), verbose = False, callbacks=[overfitCallback])
        
        else: # train for a given number of epochs
            self.model.fit(X, y,
                epochs=self.training_epochs,
                batch_size=self.batch_size,
                shuffle=True,
                validation_data=(X, y), verbose = False, callbacks=[overfitCallback]) 
        
        return self
     
    def predict(self, X):
        return self.model.predict(X)
        
    def evaluate(self, X, y=None):
        if y is None:
            y = X
        score = self.model.evaluate(X, y)
        return score
     
        