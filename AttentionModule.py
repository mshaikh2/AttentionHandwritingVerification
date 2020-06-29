from keras import backend as K
from keras.layers import Layer,InputSpec
import keras.layers as kl
import tensorflow as tf

class SelfAttention(Layer):
    def __init__(self, ch, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.channels = ch
        self.filters_f_g = self.channels // 8
        self.filters_h = self.channels

    def build(self, input_shape):
        kernel_shape_f_g = (1, 1) + (self.channels, self.filters_f_g)
        print(kernel_shape_f_g)
        kernel_shape_h = (1, 1) + (self.channels, self.filters_h)
        self.N = input_shape[1]*input_shape[2]
        # Create a trainable weight variable for this layer:
#         self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
        self.kernel_f = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_f')
        self.kernel_g = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_g')
        self.kernel_h = self.add_weight(shape=kernel_shape_h,
                                        initializer='glorot_uniform',
                                        name='kernel_h')
        self.kernel_o = self.add_weight(shape=kernel_shape_h,
                                        initializer='glorot_uniform',
                                        name='kernel_o')
        self.bias_f = self.add_weight(shape=(self.filters_f_g,),
                                      initializer='zeros',
                                      name='bias_f')
        self.bias_g = self.add_weight(shape=(self.filters_f_g,),
                                      initializer='zeros',
                                      name='bias_g')
        self.bias_h = self.add_weight(shape=(self.filters_h,),
                                      initializer='zeros',
                                      name='bias_h')
        self.bias_o = self.add_weight(shape=(self.filters_h,),
                                        initializer='glorot_uniform',
                                        name='kernel_o')
        super(SelfAttention, self).build(input_shape)
        # Set input spec.
        self.input_spec = InputSpec(ndim=4,
                                    axes={3: input_shape[-1]})
        self.built = True


    def call(self, x):
        def hw_flatten(x):
            return kl.Reshape(target_shape = (int(x.shape[1])*int(x.shape[2]),int(x.shape[3])))(x)

        f = K.conv2d(x,
                     kernel=self.kernel_f,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        f = K.bias_add(f, self.bias_f)
        f = kl.Activation('relu')(f)
        g = K.conv2d(x,
                     kernel=self.kernel_g,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        g = K.bias_add(g, self.bias_g)
        g = kl.Activation('relu')(g)
        h = K.conv2d(x,
                     kernel=self.kernel_h,
                     strides=(1, 1), padding='same')  # [bs, h, w, c]
        h = K.bias_add(h, self.bias_h)
        h = kl.Activation('relu')(h)
        s = K.tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]
        
        beta = K.softmax(s, axis=-1)  # attention map
        
        o = K.batch_dot(beta, hw_flatten(h))  # [bs, N, C]

        o = K.reshape(o, shape=K.shape(x))  # [bs, h, w, C]
        
        o = K.conv2d(o,
                     kernel=self.kernel_o,
                     strides=(1, 1), padding='same')  # [bs, h, w, c]
        o = K.bias_add(o, self.bias_h)
        o = kl.Activation('relu')(o)
        self.out_sh = tuple(o.shape.as_list())
#         x = self.gamma * o + x

        return [x, beta]#, self.gamma]

    def compute_output_shape(self, input_shape):
        return [self.out_sh,(self.N, self.N)]#, (1,)]
    
class CrossAttention(Layer):
    def __init__(self, ch, **kwargs):
        super(CrossAttention, self).__init__(**kwargs)
        self.channels = ch
        self.filters_f_g = self.channels // 8
        self.filters_h = self.channels

    def build(self, input_shape):
#         print('ca inp_shape:',input_shape)
        kernel_shape_f_g = (1, 1) + (self.channels, self.filters_f_g)
#         print(kernel_shape_f_g)
        kernel_shape_h = (1, 1) + (self.channels, self.filters_h)
        self.N = input_shape[0][1]*input_shape[0][2]
#         print('ca_module self.N:',type(self.N))
        # Create a trainable weight variable for this layer:
#         self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
        self.kernel_f = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_f')
        self.kernel_g = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_g')
        self.kernel_h = self.add_weight(shape=kernel_shape_h,
                                        initializer='glorot_uniform',
                                        name='kernel_h')
        self.kernel_o = self.add_weight(shape=kernel_shape_h,
                                        initializer='glorot_uniform',
                                        name='kernel_o')
        self.bias_f = self.add_weight(shape=(self.filters_f_g,),
                                      initializer='zeros',
                                      name='bias_f')
        self.bias_g = self.add_weight(shape=(self.filters_f_g,),
                                      initializer='zeros',
                                      name='bias_g')
        self.bias_h = self.add_weight(shape=(self.filters_h,),
                                      initializer='zeros',
                                      name='bias_h')
        self.bias_o = self.add_weight(shape=(self.filters_h,),
                                        initializer='glorot_uniform',
                                        name='kernel_o')
        super(CrossAttention, self).build(input_shape)
        # Set input spec.
#         self.input_spec = InputSpec(ndim=4,
#                                     axes={3: input_shape[0][-1]})
#         self.built = True


    def call(self, inputs):
        l,r = inputs #l = k,v; r = q 
        def hw_flatten(x):
            return kl.Reshape(target_shape = (int(x.shape[1])*int(x.shape[2]),int(x.shape[3])))(x)

        g = K.conv2d(r,
                     kernel=self.kernel_g,
                     strides=(1, 1), padding='same')  # [bs, h, w, c'] q
        g = K.bias_add(g, self.bias_g)
        g = kl.Activation('relu')(g)
        
        f = K.conv2d(l,
                     kernel=self.kernel_f,
                     strides=(1, 1), padding='same')  # [bs, h, w, c'] k
        f = K.bias_add(f, self.bias_f)
        f = kl.Activation('relu')(f)
        
        h = K.conv2d(l,
                     kernel=self.kernel_h,
                     strides=(1, 1), padding='same')  # [bs, h, w, c] v
        h = K.bias_add(h, self.bias_h)
        h = kl.Activation('relu')(h)
        
        s = K.tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  #q.k_transpose # [bs, N, N]
        
        beta = K.softmax(s, axis=-1)  # attention map
        
        o = K.batch_dot(beta, hw_flatten(h))  # [bs, N, C]

        o = K.reshape(o, shape=K.shape(l))  # [bs, h, w, C]
        
        o = K.conv2d(o,
                     kernel=self.kernel_o,
                     strides=(1, 1), padding='same')  # [bs, h, w, c]
        o = K.bias_add(o, self.bias_o)
        o = kl.Activation('relu')(o)
        self.out_sh = tuple(o.shape.as_list())
#         l = self.gamma * o + l
#         print('ca_module l.shape:',l.shape)
        return [o, beta]#, self.gamma]

    def compute_output_shape(self, input_shape):
        return [self.out_sh,(self.N, self.N)]#, (1,)]
    



class SoftAttention(Layer):
    def __init__(self,ch,m,concat_with_x=False,aggregate=False,**kwargs):
        self.channels=int(ch)
        self.multiheads = m
        self.aggregate_channels = aggregate
        self.concat_input_with_scaled = concat_with_x
#         self.h_w=self.channels//8
        
        super(SoftAttention,self).__init__(**kwargs)

    def build(self,input_shape):
#         kernel_shape=(1, 3, 3) + (self.channels, self.h_w)
#         print(input_shape)
        self.i_shape = input_shape
#         print('self.i_shape:',self.i_shape)
        kernel_shape_conv3d = (self.channels, 3, 3) + (1, self.multiheads) # DHWC
    
        self.out_attention_maps_shape = input_shape[0:1]+(self.multiheads,)+input_shape[1:-1]
        
        if self.aggregate_channels==False:
#             print(kernel_shape_conv3d)
            self.out_features_shape = input_shape[:-1]+(input_shape[-1]+(input_shape[-1]*self.multiheads),)
        else:
            if self.concat_input_with_scaled:
                self.out_features_shape = input_shape[:-1]+(input_shape[-1]*2,)
            else:
                self.out_features_shape = input_shape
        
#         print('self.out_features_shape:',self.out_features_shape)
        self.kernel_conv3d = self.add_weight(shape=kernel_shape_conv3d,
                                        initializer='glorot_uniform',
                                        name='kernel_conv3d')
        self.bias_conv3d = self.add_weight(shape=(self.multiheads,),
                                      initializer='zeros',
                                      name='bias_conv3d')
#         print('self.out_features_shape:',self.out_features_shape)
#         print('self.out_attention_maps_shape:',self.out_attention_maps_shape)
#         self.gamma1 = self.add_weight(name='gamma1', shape=[1], initializer='zeros', trainable=True)
#         self.gamma2 = self.add_weight(name='gamma2', shape=[1], initializer='ones', trainable=True)
        super(SoftAttention, self).build(input_shape)

    def call(self, x):
#         print('input shape:',x.shape) # None, 16,12,512
        exp_x = K.expand_dims(x,axis=-1)
#         print('expanded input shape:',exp_x.shape) # N, 16, 12, 512, 1
        # filters = Attention Heads
        c3d = K.conv3d(exp_x,
                     kernel=self.kernel_conv3d,
                     strides=(1,1,self.i_shape[-1]), padding='same', data_format='channels_last')
        conv3d = K.bias_add(c3d,
                        self.bias_conv3d)
        conv3d = kl.Activation('relu')(conv3d)
#         print('c3d shape:', conv3d.shape)
#         conv3d = kl.Conv3D(padding='same',filters=self.multiheads, kernel_size=(3,3,self.i_shape[-1]), strides=(1,1,self.i_shape[-1]),kernel_initializer='he_normal',activation='relu')(exp_x)
#         print('conv3d shape:', conv3d.shape)
        conv3d = K.permute_dimensions(conv3d,pattern=(0,4,1,2,3))
#         print('conv3d shape:', conv3d.shape)
#         conv3d = K.flatten(conv3d)
        
        conv3d = K.squeeze(conv3d, axis=-1)
        conv3d = K.reshape(conv3d,shape=(-1, self.multiheads ,self.i_shape[1]*self.i_shape[2]))
#         print('conv3d shape:', conv3d.shape)
#         conv3d = K.expand_dims(conv3d,axis=1) # N, 1, 16, 12       
#         print('conv3d shape:', conv3d.shape)
        softmax_alpha = K.softmax(conv3d, axis=-1) # attention map # N, 16x12
#         print('softmax_alpha shape:', softmax_alpha.shape)
        softmax_alpha = kl.Reshape(target_shape=(self.multiheads, self.i_shape[1],self.i_shape[2]))(softmax_alpha)
#         print('softmax_alpha shape:', softmax_alpha.shape)
        
        if self.aggregate_channels==False:
            exp_softmax_alpha = K.expand_dims(softmax_alpha, axis=-1) # for elementwise multiplication
    #         print('exp_softmax_alpha shape:', exp_softmax_alpha.shape)       
            exp_softmax_alpha = K.permute_dimensions(exp_softmax_alpha,pattern=(0,2,3,1,4))
    #         print('exp_softmax_alpha shape:', exp_softmax_alpha.shape)
            x_exp = K.expand_dims(x,axis=-2)
    #         print('x_exp shape:', x_exp.shape)   
            u = kl.Multiply()([exp_softmax_alpha, x_exp])   
#             print('u shape:', u.shape)   
            u = kl.Reshape(target_shape=(self.i_shape[1],self.i_shape[2],u.shape[-1]*u.shape[-2]))(u)
#             print('u shape:', u.shape)
        else:
            exp_softmax_alpha = K.permute_dimensions(softmax_alpha,pattern=(0,2,3,1))
#             print('exp_softmax_alpha shape:', exp_softmax_alpha.shape)
            exp_softmax_alpha = K.sum(exp_softmax_alpha,axis=-1)
#             print('exp_softmax_alpha after sum shape:', exp_softmax_alpha.shape)
            exp_softmax_alpha = K.expand_dims(exp_softmax_alpha, axis=-1)
#             print('exp_softmax_alpha shape:', exp_softmax_alpha.shape)
            u = kl.Multiply()([exp_softmax_alpha, x])   
#             print('u shape:', u.shape) 
        if self.concat_input_with_scaled:
            o = kl.Concatenate(axis=-1)([u,x])
        else:
            o = u
#         print('o shape:', o.shape)  
#         u = kl.Conv2D(activation='relu',filters=self.i_shape[-1],kernel_size=(1,1),padding='valid')(u)
#         u = self.gamma2 * u + x
#         print('u shape:', u.shape)
#         u = self.gamma2 * u
#         u = K.tanh(u)
#         print('u shape:', u.shape)
#         return [u, softmax_alpha]
#         self.out_features_shape = tuple(u.shape.as_list())
#         self.out_attention_maps_shape = tuple(softmax_alpha.shape.as_list())
#         print(self.out_features_shape, self.out_attention_maps_shape)
        
        return [o, softmax_alpha]

    def compute_output_shape(self, input_shape): 
        return [self.out_features_shape, self.out_attention_maps_shape]
#         return [input_shape,(None,16,12),(None,1)]
    
    def get_config(self):
        return super(SoftAttention,self).get_config()
    
class ResidualCombine2D(Layer):
    '''
    [previous_layer, concatenated_,multiattended_output]
    Combine multple heads and concat with previous layer
    Use scalar gamma for weighted concatenation
    gamma1=>residual
    gamma2=>attended
    '''
    def __init__(self, ch_in, ch_out, **kwargs):
        super(ResidualCombine2D, self).__init__(**kwargs)        
        self.image_channels = ch_in 
        self.filters_o = ch_out
    def build(self, input_shape):
        kernel_shape_o = (1,1) + (self.image_channels, self.filters_o)        
        self.gamma1 = self.add_weight(name='gamma1', shape=[1], initializer='ones', trainable=True)
        self.gamma2 = self.add_weight(name='gamma2', shape=[1], initializer='ones', trainable=True)        
        self.kernel_o = self.add_weight(shape=kernel_shape_o,
                                        initializer='glorot_uniform',
                                        name='kernel_o', trainable=True)
        self.bias_o = self.add_weight(shape=(self.filters_o,),
                                      initializer='zeros',
                                      name='bias_o', trainable=True)
        super(ResidualCombine2D, self).build(input_shape)
    def call(self, inputs):
        prev_layer, multihead_attended = inputs
        o = K.conv2d(multihead_attended,
                     kernel=self.kernel_o,
                     strides=(1,1), padding='same')
        o = K.bias_add(o, self.bias_o)
        o = kl.Activation('relu')(o)
#         print(o.shape)
        prev_layer = self.gamma1 * prev_layer 
#         print(prev_layer.shape)
#         print('x_text.shape:',x_text,x_text.shape)
        multihead_attended = self.gamma2 * o
#         print(multihead_attended.shape)
#         print('x_att.shape:',x_att,x_att.shape)
        x_out = K.concatenate([prev_layer,multihead_attended],axis=-1)
        x_out = kl.Activation('relu')(x_out)
        self.out_sh = tuple(x_out.shape.as_list())
        return [x_out, self.gamma1, self.gamma2]
    def compute_output_shape(self, input_shape):
        return [self.out_sh, tuple(self.gamma1.shape.as_list()), tuple(self.gamma2.shape.as_list())]
