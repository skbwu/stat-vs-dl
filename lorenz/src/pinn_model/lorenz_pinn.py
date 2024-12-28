import os
import time
import datetime
import math
import numpy as np
import tensorflow as tf
from decimal import Decimal
from scipy import interpolate
from src import DenseBlock

# main class of their PINN model.
class LorenzPINN(tf.keras.Model):
    
    # constructor -- just inherits keras Model.
    def __init__(self, layers=4, layer_width=20, bn=False, log_opt=False, lr=1e-2, lmbda=10.0):
        
        # initialize
        super(LorenzPINN, self).__init__()

        # sigma, rho, beta in log-scale.
        self.c1 = self.add_weight(shape=(), trainable=True, name="log-sigma", initializer=tf.keras.initializers.Constant(1.0))
        self.c2 = self.add_weight(shape=(), trainable=True, name="log-rho", initializer=tf.keras.initializers.Constant(1.0))
        self.c3 = self.add_weight(shape=(), trainable=True, name="log-beta", initializer=tf.keras.initializers.Constant(1.0))

        # builds their fully-connected layers
        self.NN = DenseBlock(layers, layer_width, bn)
        
        # internal logging of no. of epochs trained
        self.epochs = 0
        
        # are we doing optimization in log-scale or not?
        self.log_opt = log_opt
        
        # what's our optimizer?
        self.optimizer = tf.optimizers.Adam(learning_rate=lr)

        # lmbda is the tradeoff between reconstruction error and physics error
        self.lmbda = lmbda # we'll try 0.1, 1.0, 10.0, 100.0, 1000.0


    # this is their forward-pass function
    # @tf.function
    def call(self, t, TT, TM, TFC, is_forecasting):
        
        # get our rho, sigma, beta that we optimize in log-space
        theta_hat = self.predict()
        
        # let's get our x, y, z on the in-sample t that's passed in (for use on in-sample recons. error)
        [x, y, z] = self.NN(t)
        
        '''
        Note: "is_forecasting" is some vestigial setings for experimenting with other forecasting routines for PINN.
        - Such other forecasting routines perform no better than the routine presented in our paper, which
        is also the one most-commonly used in the literature.
        '''
        # if we are doing the forecasting phase, our t_physics is ONLY the FORECASTING REGION!
        if is_forecasting:
            t_physics = tf.convert_to_tensor(np.linspace(start=TM, stop=TM+TFC, 
                                                         num=math.ceil(TFC*40)+1)[1:].reshape(-1, 1))
        
        # if we're still in the in-sample training phase, our t_physics is IN-SAMPLE ONLY! 40x evenly-spaced pts / unit time.
        else:
            t_physics = tf.convert_to_tensor(np.linspace(start=TT, stop=TM, 
                                                         num=math.ceil((TM-TT)*40)+1).reshape(-1, 1))
            
        # this is the critical component where we must use automatic differentiation!
        with tf.GradientTape(persistent=True) as g:
            g.watch(t_physics)
            [x_physics, y_physics, z_physics] = self.NN(t_physics)

        # gradients w.r.t. t_physics
        dx_dt = g.gradient(x_physics, t_physics)
        dy_dt = g.gradient(y_physics, t_physics)
        dz_dt = g.gradient(z_physics, t_physics)
        
        # note for Prof. Yang -- this seems odd.
        del g

        # calculation of residual between physics-error of autodiff vs. Lorenz system formulas using in-interval
        fx = tf.cast(dx_dt, tf.float32) - theta_hat[0] * (y_physics - x_physics)
        fy = tf.cast(dy_dt, tf.float32) - x_physics * (theta_hat[1] - z_physics) + y_physics
        fz = tf.cast(dz_dt, tf.float32) - x_physics * y_physics + theta_hat[2] * z_physics

        # note that x, y, z are in-sample, while fx, fy, and fz are in-interval.
        return [x, y, z, fx, fy, fz]

    
    # just instantiating our learning rate + optimizer
    def set_lr(self, lr):
        self.optimizer = tf.optimizers.Adam(learning_rate=lr)

    # passes into the __mse function
    def get_loss(self, t, TT, TM, TFC, is_forecasting, u_true, active_comps):
        return self.__mse(u_true=u_true, 
                          y_pred=self(t=t, TT=TT, TM=TM, TFC=TFC, is_forecasting=is_forecasting), 
                          active_comps=active_comps)
    
    # this is for checking the parameter recovery only!
    def get_error(self, true):
        pred = tf.squeeze(self.predict())
        true = tf.convert_to_tensor(true)
        return tf.reduce_sum(tf.abs(pred - true))

    # converts our current conceptions of log-sigma, log-rho, log-beta to exponentiated form if necessary
    def predict(self):
        
        # convert to tensor
        var_tensor_mu = tf.convert_to_tensor([self.c1.value, self.c2.value, self.c3.value])
        
        # exponentiate
        exp_var_tensor_mu = tf.exp(var_tensor_mu) if self.log_opt else var_tensor_mu

        return exp_var_tensor_mu

    # generating trajectory reconstruction
    def predict_curves(self, t):
        return self.NN(t)
    
    # minimize our loss function which is our physics + reconstruction loss weighted sum.
    def optimize(self, t, TT, TM, TFC, is_forecasting, u_true, active_comps):
        
        # convert all arguments to tensors
        t = tf.convert_to_tensor(t, dtype=tf.float32)
        TT = tf.convert_to_tensor(TT, dtype=tf.float32)
        TM = tf.convert_to_tensor(TM, dtype=tf.float32)
        TFC = tf.convert_to_tensor(TFC, dtype=tf.float32)
        u_true = tf.convert_to_tensor(u_true, dtype=tf.float32)
        
        # start tracking gradient on all parties involved.
        with tf.GradientTape() as tape:
            loss_value = self.get_loss(t=t, TT=TT, TM=TM, TFC=TFC, is_forecasting=is_forecasting, 
                                       u_true=u_true, active_comps=active_comps)
            
        # which variables are we actually gonna take gradient step on? avoid theta if forecasting.
        trainable_vars = self.trainable_weights[:-3] if is_forecasting else self.trainable_weights
        
        # compute gradients + apply gradients to do weight update step
        grads = tape.gradient(loss_value, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        
    
    # the main function called by our pinn runscript.
    def fit(self, observed_data, TT, TM, TFC, is_forecasting, epochs, verbose=False, active_comps="XYZ"):
        
        # for each epoch ... note that we will ALWAYS have epochs = 1 to stay safe!
        for ep in range(self.epochs+1,self.epochs+epochs+1):
            
            # self.optimize calls self.get_loss, which itself is just a wrapper effectively for self.__mse
            self.optimize(observed_data[0], TT, TM, TFC, is_forecasting,
                          [observed_data[1],observed_data[2],observed_data[3]], active_comps)
        
        # this is an INTERNAL self-counter! we will use the for-loop in our main runner script.
        self.epochs += epochs

    
    # this COMPUTES OUR WEIGHTED RECONSTRUCTION + PHYSICS ERRORS!
    def __mse(self, u_true, y_pred, active_comps):
        
        # let's assemble our reconstruction loss sequentially
        recons_loss = 0.0
        
        '''
        To account for missing components and avoid NaNs.
        '''
        # add L2 component loss components only if they are observed!
        if "X" in active_comps:
            loss_x = tf.reduce_mean(tf.square(y_pred[0] - u_true[0]))
            recons_loss += (self.lmbda*loss_x)
        if "Y" in active_comps:
            loss_y = tf.reduce_mean(tf.square(y_pred[1] - u_true[1]))
            recons_loss += (self.lmbda*loss_y)
        if "Z" in active_comps:
            loss_z = tf.reduce_mean(tf.square(y_pred[2] - u_true[2]))
            recons_loss += (self.lmbda*loss_z)
        
        # always compute + add the physics-error residuals!
        loss_fx = tf.reduce_mean(tf.square(y_pred[3]))
        loss_fy = tf.reduce_mean(tf.square(y_pred[4]))
        loss_fz = tf.reduce_mean(tf.square(y_pred[5]))
        
        # already-weighted sum of L2 loss & physics-error residuals
        return recons_loss + (loss_fx + loss_fy + loss_fz)