import numpy as np
from .functions import activate, deactivate

class FFL():
    """
        A class to create a single input/hidden feedforward layer of given shape.
        
        Input:
        -------
        n_input: Number of input values to layer. If none, tried to check from preious layer.
        neurons: Neurons on that layer.
        bias: External bias numpy array. If None, used from np.random.randn(neurons)
        weights: External weight numpy array. If None, used from np.random.randn(n_input, neurons)
        activation: One of ["relu", "sigmoid", "tanh", "softmax"].
        is_bias: Do we want bias to be used here? Default True.
        
        Output:
        --------
        Object of FFL which can be stacked later to use.
        
        Example:
        FFL(2, 2, activation="sigmoid")
    """
    
    def __init__(self, input_shape=None, neurons=1, bias=None, weights=None, activation=None, is_bias = True):
        
        np.random.seed(100)
        self.input_shape = input_shape
        self.neurons = neurons
        self.isbias = is_bias
        self.name = ""
        
        
        self.w = weights
        self.b = bias
        if input_shape != None:
            self.output_shape = neurons
        
        
        if self.input_shape != None:
            self.weights = weights if weights != None else np.random.randn(self.input_shape, neurons)
            self.parameters = self.input_shape *  self.neurons + self.neurons if self.isbias else 0  
        if(is_bias):
            self.biases = bias if bias != None else np.random.randn(neurons)
        else:
            self.biases = 0
            
        self.out = None
        self.input = None
        self.error = None
        self.delta = None
        activations = ["linear", "relu", "sigmoid", "tanh", "softmax"]
        self.delta_weights = 0
        self.delta_biases = 0
        self.pdelta_weights = 0
        self.pdelta_biases = 0
        
        #self.bias = np.ones(self.bias.shape)
        
        if activation not in activations and activation != None:
             raise ValueError(f"Activation function not recognised. Use one of {activations} instead.")
        else:
            self.activation = activation

        if self.activation == None:
            self.activation = "linear"
        

    def activation_dfn(self, r):
        """
            A method of FFL to find derivative of given activation function.
        """
        a = deactivate(self.activation, r)
        return a
        # if self.activation == "relu":
        #     r[r<0] = 0
        #     return r
        
        # if self.activation is None:
        #     return np.ones(r.shape)
        
        # if self.activation == 'tanh':
        #     return 1 - r ** 2

        # if self.activation == 'sigmoid':
        #     return r * (1 - r)

        # if self.activation == "softmax":
        #     soft = self.activation_fn(r)
                    
        #     #s = soft.reshape(-1, 1)
        #     #dsoft = np.diagflat(s) - np.dot(s, s.T)
        #     #diag_soft = dsoft.diagonal()
            
        #     # take only the diagonal of dsoft i.e i==j only
        #     """
        #         soft = a / np.sum(a)

        #         dsoft = np.diag(soft)
        #         for i in range(len(x)):
        #             for j in range(len(soft)):
        #                 if i == j:
        #                     d = 1
        #                 else:
        #                     d = 0
        #                 dsoft[i][j] = soft[i] * (d - soft[j])
        #     """
        #     diag_soft = soft*(1- soft)
        #     return diag_soft
        
        # return r

    def activation_fn(self, r):
        """
        A method of FFL which contains the operation and defination of given activation function.
        """
        a = activate(self.activation, r)
        return a
        
        # if self.activation == None or self.activation == "linear":
        #     return r

        # # tanh
        # if self.activation == 'tanh':
        #     return np.tanh(r)

        # # sigmoid
        # if self.activation == 'sigmoid':
            
        #     return 1 / (1 + np.exp(-r))

        # if self.activation == "softmax":
        #     # stable softmax
        #     r = r - np.max(r)
        #     s = np.exp(r)
        #     return s / np.sum(s)
        # if self.activation == "relu":
        #     r[r<0] = 0
        #     return r
        
    def apply_activation(self, x):
        soma = np.dot(x, self.weights) + self.biases
        self.out = self.activation_fn(soma)        
        return self.out

    def set_n_input(self):
        self.weights = self.w if self.w != None else np.random.normal(size=(self.input_shape, self.neurons))
    def backpropagate(self, nx_layer):
        self.error = np.dot(nx_layer.weights, nx_layer.delta)
        self.delta = self.error * self.activation_dfn(self.out)
        self.delta_weights += self.delta * np.atleast_2d(self.input).T
        self.delta_biases += self.delta
    
    # added below methods from cnn
    def set_output_shape(self):
        self.set_n_input()
        self.output_shape = self.neurons
        self.get_parameters()
    def get_parameters(self):
        self.parameters = self.input_shape *  self.neurons + self.neurons if self.isbias else 0  
        return self.parameters

class Dropout:
    def __init__(self, prob = 0.5):
        self.input_shape=None
        self.output_shape = None
        self.input_data= None
        self.output = None
        self.isbias = False
        self.activation = None
        self.parameters = 0
        self.delta = 0
        self.weights = 0
        self.bias = 0
        self.prob = prob
        self.delta_weights = 0
        self.delta_biases = 0
        
    def set_output_shape(self):
        #print(self.input_shape)
        self.output_shape = self.input_shape
        self.weights = 0
    def apply_activation(self, x, train=True):
        if train:
            self.input_data = x
            #print(x.sum(axis=2))
            flat = np.array(self.input_data).flatten()
            random_indices = np.random.randint(0, len(flat), int(self.prob * len(flat)))
            flat[random_indices] = 0
            self.output = flat.reshape(x.shape)
            return self.output
        else:
            self.input_data = x
            self.output = x / self.prob
            return self.output
    def activation_dfn(self, x):
        return x
    def backpropagate(self, nx_layer):
        if type(nx_layer).__name__ != "Conv2d":
            self.error = np.dot(nx_layer.weights, nx_layer.delta)
            self.delta = self.error * self.activation_dfn(self.out)
        else:
            self.delta = nx_layer.delta
        self.delta[self.output == 0] = 0

class Pool2d:
    def __init__(self, kernel_size = (2, 2), stride=None, kind="max", padding=None):
        self.input_shape=None
        self.output_shape = None
        self.input_data= None
        self.output = None
        self.isbias = False
        self.activation = None
        self.parameters = 0
        self.delta = 0
        self.weights = 0
        self.bias = 0
        self.delta_weights = 0
        self.delta_biases = 0
        self.padding = padding
        self.p = 1 if padding != None else 0
        self.kernel_size = kernel_size
        if type(stride) == int:
                 stride = (stride, stride)

        self.stride = stride
        if self.stride == None:
            self.stride = self.kernel_size
        self.pools = ['max', "average", 'min']
        if kind not in self.pools:
            raise ValueError("Pool kind not understoood.")
            
        self.kind = kind
    
    def set_output_shape(self):
        shape = self.input_shape
        self.output_shape = (int((self.input_shape[0] - self.kernel_size[0] + 2 * self.p) / self.stride[0] + 1), 
                                int((self.input_shape[1] - self.kernel_size[1] + 2 * self.p) / self.stride[1] + 1), self.input_shape[2])
        

        self.weights = 0
    def apply_activation(self, image):
        stride = self.stride
        kshape = self.kernel_size
        shape = image.shape
        self.input_shape = shape
        self.set_output_shape()
        #print(self.output_shape, shape)
        
        
        rstep = stride[0]+kshape[0]-1
        cstep = stride[1]+kshape[1]-1
        self.out = np.zeros((self.output_shape))
        for nc in range(shape[2]):
            cimg = []
            rv = 0
            for r in range(kshape[0], shape[0]+1, stride[0]):
                cv = 0
                for c in range(kshape[1], shape[1]+1, stride[1]):
                    chunk = image[rv:r, cv:c, nc]
                    if len(chunk) > 0:
                        
                        if self.kind == "max":
                            #print(chunk)
                            chunk = np.max(chunk)
                        if self.kind == "min":
                            chunk = np.min(chunk)
                        if self.kind == "average":
                            chunk = np.mean(chunk)

                        cimg.append(chunk)
                    else:
                        cv-=cstep
                        #rv-=rstep
                    cv+=stride[1]
                rv+=stride[0]
            cimg = np.array(cimg).reshape(int(rv/stride[0]), int(cv/stride[1]))
            #print(cimg.shape)
            self.out[:,:,nc] = cimg
        return self.out
    def backpropagate(self, nx_layer):
        """
            Gradients are passed through index of largest value .
        """
        layer = self
        stride = layer.stride
        kshape = layer.kernel_size
        image = layer.input
        shape = image.shape
        layer.delta = np.zeros(shape)
        #self.input_shape = shape
        #self.set_output_shape()
        #print(self.output_shape)
        
        cimg = []
        rstep = stride[0]
        cstep = stride[1]
        
        for f in range(shape[2]):
            i = 0
            rv = 0
            for r in range(kshape[0], shape[0]+1, rstep):
                cv = 0
                j = 0
                for c in range(kshape[1], shape[1]+1, cstep):
                    chunk = image[rv:r, cv:c, f]
                    dout = nx_layer.delta[i, j, f]
                    
                    if layer.kind == "max":
                        p = np.max(chunk)
                        #print(p, chunk)
                        index = np.argwhere(chunk == p)[0]
                        #print(layer.delta[rv+index[0], cv+index[1], f].shape, dout.shape)
                        layer.delta[rv+index[0], cv+index[1], f] = dout
                        #print(index)
                    if layer.kind == "min":
                        p = np.min(chunk)
                        index = np.argwhere(chunk == p)[0]
                        layer.delta[rv+index[0], cv+index[1], f] = dout
                    if layer.kind == "average":
                        p = np.mean(chunk)
                        layer.delta[rv:r, cv:c, f] = dout


                    j+=1
                    cv+=cstep
                rv+=rstep
                i+=1
            
class Flatten:
    def __init__(self, input_shape=None):
        self.input_shape=None
        self.output_shape = None
        self.input_data= None
        self.output = None
        self.isbias = False
        self.activation = None
        self.parameters = 0
        self.delta = 0
        self.weights = 0
        self.bias = 0
        self.delta_weights = 0
        self.delta_biases = 0
        
    def set_output_shape(self):
        #print(self.input_shape)
        self.output_shape = (self.input_shape[0] * self.input_shape[1] * self.input_shape[2])
        self.weights = 0
    def apply_activation(self, x):
        self.input_data = x
        #print(x.sum(axis=2))
        self.output = np.array(self.input_data).flatten()
        return self.output
    def activation_dfn(self, x):
        return x
    def backpropagate(self, nx_layer):
        self.error = np.dot(nx_layer.weights, nx_layer.delta)
        self.delta = self.error * self.activation_dfn(self.out)
        self.delta = self.delta.reshape(self.input_shape)

class Conv2d():
    def __init__(self, input_shape=None, filters=1, kernel_size = (3, 3), isbias=True, activation=None, stride=(1, 1), padding="zero", kernel=None, bias=None):
        #super().__init__()
        self.input_shape = input_shape
        self.filters = filters
        self.isbias = isbias
        self.activation = activation
        self.stride = stride
        self.padding = padding
        self.p = 1 if padding != None else 0
        self.bias = bias
        self.kernel = kernel
        if input_shape != None:
            # (h, w, nc, nf)
            self.kernel_size = (kernel_size[0], kernel_size[1], input_shape[2], filters)
            self.output_shape = (int((input_shape[0] - kernel_size[0] + 2 * self.p) / stride[0]) + 1, 
                                int((input_shape[1] - kernel_size[1] + 2 * self.p) / stride[1]) + 1, filters)
            self.set_variables()
            self.out = np.zeros(self.output_shape)
            self.dout = np.zeros((self.output_shape[0], self.output_shape[1], self.input_shape[2], self.output_shape[2]))
        else:
            #pass
            # we dont know input shape yet
            self.kernel_size = (kernel_size[0], kernel_size[1])
            #self.set_variables()
            
        self.delta = 0
        self.dfilt = 0
        self.dbias = 0
        self.error = 0
        #self.out = np.zeros(self.output_shape)
        #self.dout = np.zeros(self.output_shape)
        if self.activation == None:
            self.activation = "linear"
        
        # make suitable for all layers
        #self.weights = self.kernel
        #self.biases = self.bias
        
    def init_param(self, size):
        stddev = 1/np.sqrt(np.prod(size))
        return np.random.normal(loc=0, scale=stddev, size=size)
    def set_variables(self):
        """
        self.bias = np.random.random((self.filters, 1))
        #self.bias = np.random.random((self.kernel_size[0], self.kernel_size[1], self.input_shape[2], self.filters)) if self.isbias == True and self.bias==None else 0
        self.kernel = np.random.random((self.kernel_size[0], self.kernel_size[1], self.input_shape[2], self.filters)) if self.kernel == None else self.kernel
        self.parameters = np.multiply.reduce(self.kernel_size) * 2 if self.isbias else 1        
        self.d_final_filt = np.zeros(self.kernel.shape)
        self.d_final_bias = np.zeros(self.bias.shape)
        self.kernel = self.init_param(self.kernel.shape)
        """
        self.weights = self.init_param(self.kernel_size)
        self.biases = self.init_param((self.filters, 1))
        self.parameters = np.multiply.reduce(self.kernel_size) + self.filters if self.isbias else 1
        self.delta_weights = np.zeros(self.kernel_size)
        self.delta_biases = np.zeros(self.biases.shape)
        #print(self.weights.shape, self.biases.shape)
       
    def apply_activation(self, image):
        #print(self.weights, self.biases)
        for f in range(self.filters):
            image = self.input
            kshape = self.kernel_size
            if kshape[0] % 2 != 1 or kshape[1] % 2 != 1:
                raise ValueError("Please provide odd length of 2d kernel.")

            if type(self.stride) == int:
                     stride = (stride, stride)
            else:
                stride = self.stride
            shape = image.shape
            if self.padding == "zero":
                zeros_h = np.zeros((shape[1], shape[2])).reshape(-1, shape[1], shape[2])
                zeros_v = np.zeros((shape[0]+2, shape[2])).reshape(shape[0]+2, -1, shape[2])
                #print(image.shape, zeros_h.shape, zeros_v.shape)
                #zero padding
                padded_img = np.vstack((zeros_h, image, zeros_h)) # add rows
                # print(padded_img)
                padded_img = np.hstack((zeros_v, padded_img, zeros_v)) # add cols

                image = padded_img
                shape = image.shape

            elif self.padding == "same":
                h1 = image[0].reshape(-1, shape[1], shape[2])
                h2 = image[-1].reshape(-1, shape[1], shape[2])


                #zero padding
                padded_img = np.vstack((h1, image, h2)) # add rows

                v1 = padded_img[:, 0].reshape(padded_img.shape[0], -1, shape[2])
                v2 = padded_img[:, -1].reshape(padded_img.shape[0], -1, shape[2])

                padded_img = np.hstack((v1, padded_img, v2)) # add cols

                image = padded_img
                shape = image.shape
            elif self.padding == None:
                pass

            rv = 0
            cimg = []
            for r in range(kshape[0], shape[0]+1, stride[0]):
                cv = 0
                for c in range(kshape[1], shape[1]+1, stride[1]):
                    #if r - rv
                    chunk = image[rv:r, cv:c]
                    #print(chunk.shape, self.kernel[:, :, :, f].shape)
                    # soma with each input image's chunk(can have channel) * kernel(can have channels)
                    soma = (np.multiply(chunk, self.weights[:, :, :, f]))
                    summa = soma.sum()+self.biases[f]
                    #print(soma.shape)
                    #summa = soma.mean()
                    #print(f"soma{soma.shape} \n chunk{chunk.shape}, {summa.shape}")
                    
                    cimg.append(summa)
                    cv+=stride[1]
                rv+=stride[0]
            #print((r, c), (rv, cv))
            cimg = np.array(cimg).reshape(int(rv/stride[0]), int(cv/stride[1]))
            #print(cimg.shape)
            self.out[:, :, f] = cimg
        self.out = self.activation_fn(self.out)
        return self.out
        
    
    def activation_dfn(self, r):
        """
            A method of FFL to find derivative of given activation function.
        """
        a = deactivate(self.activation, r)
        return a
        
        # if self.activation is None:
        #     return np.ones(r.shape)

        # if self.activation == 'tanh':
        #     return 1 - r ** 2
        # if self.activation == "relu":
        #     r[r<0] = 0
        #     return r
        # if self.activation == 'sigmoid':
        #     return r * (1 - r)

        # if self.activation == "softmax":
        #     soft = self.activation_fn(r)
        #     diag_soft = soft * (1-soft)        
        #     #s = soft.reshape(-1, 1)
        #     #dsoft = np.diagflat(s) - np.dot(s, s.T)
        #     #diag_soft = dsoft.diagonal()
        #     # take only the diagonal of dsoft i.e i==j only
        #     """
        #         soft = a / np.sum(a)

        #         dsoft = np.diag(soft)
        #         for i in range(len(x)):
        #             for j in range(len(soft)):
        #                 if i == j:
        #                     d = 1
        #                 else:
        #                     d = 0
        #                 dsoft[i][j] = soft[i] * (d - soft[j])
        #     """
            
        #     return diag_soft
        
        # return r
    
    def activation_fn(self, r):
        """
        A method of FFL which contains the operation and defination of given activation function.
        """
        a = activate(self.activation, r)
        return a
        
        # if self.activation == None or self.activation == "linear":
        #     return r
        # if self.activation == "relu":
        #     r[r<0] = 0
        #     return r
        
        # # tanh
        # if self.activation == 'tanh':
        #     return np.tanh(r)

        # # sigmoid
        # if self.activation == 'sigmoid':
            
        #     return 1 / (1 + np.exp(-r))

        # if self.activation == "softmax":
        #     # stable softmax
        #     r = r - np.max(r)
        #     s = np.exp(r)
        #     return s / np.sum(s)
        
    def set_output_shape(self):
        #print(self.input_shape, self.kernel_size, self.stride)
        self.kernel_size = (self.kernel_size[0], self.kernel_size[1], self.input_shape[2], self.filters)
        self.set_variables()
        self.output_shape = (int((self.input_shape[0] - self.kernel_size[0] + 2 * self.p) / self.stride[0] + 1), 
                                int((self.input_shape[1] - self.kernel_size[1] + 2 * self.p) / self.stride[1] + 1), self.filters)
        self.out = np.zeros(self.output_shape)
        self.dout = np.zeros((self.output_shape[0], self.output_shape[1], self.input_shape[2], self.output_shape[2]))
    
    def backpropagate(self, nx_layer):
        layer = self
        #layer.dfilt = np.zeros(layer.kernel.shape)
        #layer.delta = np.zeros(layer.input_shape)
        #layer.dbias = np.zeros(layer.bias.shape)
        layer.delta = np.zeros((layer.input_shape[0], layer.input_shape[1], layer.input_shape[2]))
        
        image = layer.input
        #print(layer.name, layer.delta.shape)
        #print(f"nx lyr's delshape", nx_layer.delta.shape, layer.input_shape)
        #print(f"{nx_layer.delta.shape, layer.kernel.shape}")
         
        for f in range(layer.filters):
            kshape = layer.kernel_size
            shape = layer.input_shape
            stride = layer.stride
            rv = 0
            i = 0
            #print(layer.kernel.shape)
            for r in range(kshape[0], shape[0]+1, stride[0]):
                cv = 0
                j = 0
                for c in range(kshape[1], shape[1]+1, stride[1]):
                    chunk = image[rv:r, cv:c]
                    #print(layer.name, chunk.shape, layer.dfilt.shape, nx_layer.delta.shape)
                    #"""
                    #original
                    #print(layer.delta_weights.shape)
                    layer.delta_weights[:, :, :, f] += chunk * nx_layer.delta[i, j, f]
                    
                    #layer.delta[rv:r, cv:c] += nx_layer.delta[i, j] * layer.dfilt[f]
                    #print("upd", layer.delta[rv:r, cv:c, f].shape, nx_layer.delta[i, j, f].shape, layer.kernel[:, :, :, f].shape)
                    #print(layer.name, layer.kernel.shape)
                    layer.delta[rv:r, cv:c, :] += nx_layer.delta[i, j, f] * layer.weights[:, :, :, f]
                    #chunk = int((np.multiply(chunk, kernel)+bias).sum())
                    j+=1
                    cv+=stride[1]
                rv+=stride[0]
                i+=1
            #layer.kernel[f] += dfilt
            layer.delta_biases[f] = np.sum(nx_layer.delta[:, :, f])
        layer.delta = layer.activation_dfn(layer.delta)
