import os
import time
import datetime
import math
import numpy as np
import tensorflow as tf
from decimal import Decimal
from scipy import interpolate
from pinn.src import DenseBlock

# main class of their PINN model.
class SEIRPINN_v2(tf.keras.Model):
    
    # constructor -- just inherits keras Model.
    def __init__(self, layers=4, layer_width=20, bn=False, log_opt=False, lr=1e-2, lmbda=10.0):
        
        # initialize
        super(SEIRPINN_v2, self).__init__()

        # beta & gamma on the log-scale
        self.log_beta = self.add_weight(shape=(), trainable=True, name="log-beta",
                                        initializer=tf.keras.initializers.Constant(1.0))
        self.log_gamma = self.add_weight(shape=(), trainable=True, name="log-gamma", 
                                         initializer=tf.keras.initializers.Constant(1.0))
        self.log_sigma = self.add_weight(shape=(), trainable=True, name="log-sigma", 
                                         initializer=tf.keras.initializers.Constant(1.0))

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
        
        # get our (beta, gamma) that we optimized in log-space
        beta, gamma, sigma = tf.exp(self.log_beta), tf.exp(self.log_gamma), tf.exp(self.log_sigma)
        
        # let's get our e, i, r on the in-sample t that's passed in (for use on in-sample recons. error)
        [E_log, I_log, R_log] = self.NN(t)
        
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
            [E_log_physics, I_log_physics, R_log_physics] = self.NN(t_physics)

        dE_log_dt = g.gradient(E_log_physics, t_physics)
        dI_log_dt = g.gradient(I_log_physics, t_physics)
        dR_log_dt = g.gradient(R_log_physics, t_physics)

        # Compute E, I, R from their logs
        E_physics = tf.exp(E_log_physics)
        I_physics = tf.exp(I_log_physics)
        R_physics = tf.exp(R_log_physics)
        S_physics = 1.0 - E_physics - I_physics - R_physics
        del g

        # Theoretical derivatives from log-form equations:
        dE_log_dt_theory = beta * S_physics * tf.exp(I_log_physics - E_log_physics) - sigma
        dI_log_dt_theory = sigma * tf.exp(E_log_physics - I_log_physics) - gamma
        dR_log_dt_theory = gamma * tf.exp(I_log_physics - R_log_physics)

        # Residuals for the PDE in log form:
        fE = tf.cast(dE_log_dt, tf.float32) - dE_log_dt_theory
        fI = tf.cast(dI_log_dt, tf.float32) - dI_log_dt_theory
        fR = tf.cast(dR_log_dt, tf.float32) - dR_log_dt_theory

        # note that x, y, z are in-sample, while fx, fy, and fz are in-interval.
        return [E_log, I_log, R_log, fE, fI, fR]

    
    # just instantiating our learning rate + optimizer
    def set_lr(self, lr):
        self.optimizer = tf.optimizers.Adam(learning_rate=lr)

        
    # passes into the __mse function
    def get_loss(self, t, TT, TM, TFC, is_forecasting, u_true, active_comps):
        return self.__mse(u_true=u_true, 
                          y_pred=self(t=t, TT=TT, TM=TM, TFC=TFC, is_forecasting=is_forecasting), 
                          active_comps=active_comps)
    
    
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
    def fit(self, observed_data, TT, TM, TFC, is_forecasting, epochs, verbose=False, active_comps="SIR"):
        
        # for each epoch ... note that we will ALWAYS have epochs = 1 to stay safe!
        for ep in range(self.epochs+1,self.epochs+epochs+1):
            
            # self.optimize calls self.get_loss, which itself is just a wrapper effectively for self.__mse
            self.optimize(observed_data[0], TT, TM, TFC, is_forecasting,
                          [observed_data[1], observed_data[2], observed_data[3]], active_comps)
        
        # this is an INTERNAL self-counter! we will use the for-loop in our main runner script.
        self.epochs += epochs

    
    # this COMPUTES OUR WEIGHTED RECONSTRUCTION + PHYSICS ERRORS!
    def __mse(self, u_true, y_pred, active_comps):
        
        # let's assemble our reconstruction loss sequentially
        recons_loss = 0.0
        
        # add L2 component loss components only if they are observed!
        if "E" in active_comps:
            loss_e = tf.reduce_mean(tf.square(y_pred[0] - u_true[0]))
            recons_loss += (self.lmbda * loss_e)
        if "I" in active_comps:
            loss_i = tf.reduce_mean(tf.square(y_pred[1] - u_true[1]))
            recons_loss += (self.lmbda * loss_i)
        if "R" in active_comps:
            loss_r = tf.reduce_mean(tf.square(y_pred[2] - u_true[2]))
            recons_loss += (self.lmbda * loss_r)

        # always compute + add the physics-error residuals!
        loss_fe = tf.reduce_mean(tf.square(y_pred[3]))
        loss_fi = tf.reduce_mean(tf.square(y_pred[4]))
        loss_fr = tf.reduce_mean(tf.square(y_pred[5]))

        # already-weighted sum of L2 loss & physics-error residuals
        return recons_loss + (loss_fe + loss_fi + loss_fr)
