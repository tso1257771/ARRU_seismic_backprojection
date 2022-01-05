import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax, sigmoid
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import *

class unets:
    def __init__(self, 
                input_size=(2001, 3), 
                nb_filters = [6, 12, 18, 24],
                kernel_size = 7,
                kernel_init = 'he_uniform',
                kernel_regu = tf.keras.regularizers.l1(1e-4),
                activation = 'relu',
                out_activation = 'softmax',
                dropout_rate = 0.1,
                batchnorm = True,
                max_pool = False,
                pool_size = 4,
                stride_size = 4,
                upsize = 4,
                padding = 'same',
                RRconv_time=3):

        self.input_size = input_size
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.kernel_init = kernel_init
        self.kernel_regu = kernel_regu
        self.activation = activation
        self.out_activation = out_activation
        self.dropout_rate = dropout_rate
        self.batchnorm = batchnorm
        self.max_pool = max_pool
        self.pool_size = pool_size
        self.stride_size = stride_size
        self.upsize = upsize
        self.padding = padding
        self.RRconv_time = RRconv_time
    
    def conv1d(self, nb_filter, stride_size=None):
        if stride_size:
            return Conv1D(nb_filter, self.kernel_size, 
                padding=self.padding, strides=stride_size,
                kernel_initializer=self.kernel_init, 
                kernel_regularizer=self.kernel_regu)
        else:
            return Conv1D(nb_filter, self.kernel_size, 
                padding=self.padding,
                kernel_initializer=self.kernel_init, 
                kernel_regularizer=self.kernel_regu)

    def att_block(self, xl, gate):
        # xl = input feature (U net left hand side)
        # gate = gating signal (U net right hand side)
        F_l = int(xl.shape[1])
        F_g = int(gate.shape[1])
        F_int = int(xl.shape[2])

        W_x = Conv1D(F_l, F_int, strides=1, padding=self.padding)(xl)
        W_x_n = BatchNormalization()(W_x)

        W_g = Conv1D(F_g, F_int, strides=1, padding=self.padding)(gate)
        W_g_n = BatchNormalization()(W_g)

        add = Add()([W_x_n, W_g_n])
        add = Activation('relu')(add)

        psi = Conv1D(F_int, 1, strides=1, padding=self.padding)(add)
        psi_n = BatchNormalization()(psi)
        psi_activate = Activation('sigmoid')(psi_n) 

        mul = Multiply()([xl, psi_activate])

        return mul
    # when to add B.N. layer, before or after activation layer?
    def conv_unit(self, inputs, nb_filter, stride_size):

        if self.max_pool:
            u = self.conv1d(nb_filter, self.pool_size)(inputs)
            if self.batchnorm:
                u = BatchNormalization()(u)
            u = Activation(self.activation)(u)
            if self.dropout_rate:
                u = Dropout(self.dropout_rate)(u)
            u = MaxPooling1D(pool_size=self.pool_size, padding=self.padding)(u)
        else:
            if (stride_size != None) :
                u = self.conv1d(nb_filter, stride_size=stride_size)(inputs)
                if self.batchnorm:
                    u = BatchNormalization()(u)
                u = Activation(self.activation)(u)
                if self.dropout_rate:
                    u = Dropout(self.dropout_rate)(u)
            
            else:
                u = self.conv1d(nb_filter)(inputs)
                if self.batchnorm:
                    u = BatchNormalization()(u)
                u = Activation(self.activation)(u)
                if self.dropout_rate:
                    u = Dropout(self.dropout_rate)(u)
        return u

    def RRconv_unit(self, inputs, nb_filter, stride_size):
        
        if stride_size==None:
            u = self.conv_unit(inputs=inputs, nb_filter=nb_filter, stride_size=None)
        else:
            u = self.conv_unit(inputs=inputs, nb_filter=nb_filter, stride_size=stride_size)
        conv_1x1 = self.conv1d(nb_filter=nb_filter, stride_size=1)(u)
        for i in range(self.RRconv_time):
            if i == 0:
                r_u = u
            r_u = Add()([r_u, u])
            r_u = self.conv_unit(inputs=r_u, nb_filter=nb_filter, stride_size=None)
        
        return Add()([r_u, conv_1x1])

    def upconv_unit(self, inputs, nb_filter, concatenate_layer, 
            apply_attention=False, att_transformer=False):
            # transposed convolution
        u = UpSampling1D(size=self.upsize)(inputs)
        u = self.conv1d(nb_filter, stride_size=None)(u)
        if self.batchnorm:
            u = BatchNormalization()(u)
        u = Activation(self.activation)(u)
        if self.dropout_rate:
            u = Dropout(self.dropout_rate)(u)
        #u.shape TensorShape([None, 128, 18])
        # concatenate_layer.shape TensorShape([None, 126, 18])
        shape_diff = u.shape[1] - concatenate_layer.shape[1]
        if shape_diff>0:
            crop_shape = (shape_diff//2, shape_diff-shape_diff//2)
        else:
            crop_shape = None

        if apply_attention and (att_transformer==False):
            if crop_shape:
                crop = Cropping1D(cropping=crop_shape)(u)
                att = self.att_block(xl=concatenate_layer, gate=crop)
                upconv = concatenate([att, crop])
            elif not crop_shape:
                att = self.att_block(xl=concatenate_layer, gate=u)
                upconv = concatenate([att, u])

        if apply_attention and (att_transformer==True):
            att, attW = _transformer(self.dropout_rate, width=3, name=None, 
                inpC=concatenate_layer)
            if crop_shape:
                crop = Cropping1D(cropping=crop_shape)(u)
                upconv = concatenate([att, crop])
            elif not crop_shape:
                att, attW = _transformer(self.dropout_rate, width=3, name=None, 
                    inpC=concatenate_layer)
                upconv = concatenate([att, u])

        elif not apply_attention:
            if crop_shape:
                crop = Cropping1D(cropping=crop_shape)(u)      
                upconv = concatenate([concatenate_layer, crop])
            elif not crop_shape:
                upconv = concatenate([concatenate_layer, u])            

        return upconv

    def build_unet(self, pretrained_weights=None, input_size=None, nb_filters=None):
        if nb_filters == None:
            nb_filters = self.nb_filters
        if input_size == None:
            input_size = self.input_size
        inputs = Input(input_size)
        conv_init_exp = self.conv_unit(inputs=inputs, nb_filter=nb_filters[0], stride_size=None)
        #conv_init_exp = conv_unit(inputs=conv_init_exp, nb_filter=nb_filter[0], stride_size=None)     
        # (None, 1, nb_filter[0], 2001)
        #========== Encoder
        down1 = self.conv_unit(inputs=conv_init_exp, nb_filter=nb_filters[0], stride_size=self.stride_size) # downsample
        down1_exp = self.conv_unit(inputs=down1, nb_filter=nb_filters[1], stride_size=None)
        # (None, 501, nb_filter[0])
        down2 = self.conv_unit(inputs=down1_exp, nb_filter=nb_filters[1], stride_size=self.stride_size)
        down2_exp = self.conv_unit(inputs=down2, nb_filter=nb_filters[2], stride_size=None)
        # (None, 126, nb_filter[1])
        down3 = self.conv_unit(inputs=down2_exp, nb_filter=nb_filters[2], stride_size=self.stride_size)
        down3_exp = self.conv_unit(inputs=down3, nb_filter=nb_filters[3], stride_size=None)
        # (None, 32, nb_filter[2])

        #========== Decoder
        up4 = self.upconv_unit(inputs=down3_exp, nb_filter=nb_filters[2], concatenate_layer=down2_exp)
        up4_fus = self.conv_unit(inputs=up4, nb_filter=nb_filters[2], stride_size=None)

        up5 = self.upconv_unit(inputs=up4_fus, nb_filter=nb_filters[1], concatenate_layer=down1_exp)
        up5_fus = self.conv_unit(inputs=up5, nb_filter=nb_filters[1], stride_size=None)

        up6 = self.upconv_unit(inputs=up5_fus, nb_filter=nb_filters[0], concatenate_layer=conv_init_exp)
        up6_fus = self.conv_unit(inputs=up6, nb_filter=nb_filters[0], stride_size=None)

        ##========== Output map
        outmap = Conv1D(3, 1, kernel_initializer='he_uniform', name='pred_label')(up6_fus)#, kernel_regularizer=l2(1e-3)
        outmask = Conv1D(2, 1, kernel_initializer='he_uniform', name='pred_mask')(up6_fus)
        outmap_Act = Activation(self.out_activation)(outmap)
        outmask_Act = Activation(self.out_activation)(outmask)
        model = Model(inputs=inputs, outputs=[outmap_Act, outmask_Act])

        # compile
        if pretrained_weights==None:
            return model             

        else:
            model.load_weights(pretrained_weights)    
            return model

    def build_R2unet(self, pretrained_weights=None, input_size=None, nb_filters=None):
        if nb_filters == None:
            nb_filters = self.nb_filters
        if input_size == None:
            input_size = self.input_size
        inputs = Input(input_size)
        conv_init_exp = self.RRconv_unit(inputs=inputs, nb_filter=nb_filters[0], stride_size=None)   
        # (None, 1, nb_filter[0], 2001)
        #========== Encoder
        down1 = self.RRconv_unit(inputs=conv_init_exp, nb_filter=nb_filters[0], stride_size=self.stride_size) # downsample
        down1_exp = self.RRconv_unit(inputs=down1, nb_filter=nb_filters[1], stride_size=None)
        # (None, 501, nb_filter[0])
        down2 = self.RRconv_unit(inputs=down1_exp, nb_filter=nb_filters[1], stride_size=self.stride_size)
        down2_exp = self.RRconv_unit(inputs=down2, nb_filter=nb_filters[2], stride_size=None)
        # (None, 126, nb_filter[1])
        down3 = self.RRconv_unit(inputs=down2_exp, nb_filter=nb_filters[2], stride_size=self.stride_size)
        down3_exp = self.RRconv_unit(inputs=down3, nb_filter=nb_filters[3], stride_size=None)
        # (None, 32, nb_filter[2])

        #========== Decoder
        up4 = self.upconv_unit(inputs=down3_exp, nb_filter=nb_filters[2], concatenate_layer=down2_exp)
        up4_fus = self.RRconv_unit(inputs=up4, nb_filter=nb_filters[2], stride_size=None)

        up5 = self.upconv_unit(inputs=up4_fus, nb_filter=nb_filters[1], concatenate_layer=down1_exp)
        up5_fus = self.RRconv_unit(inputs=up5, nb_filter=nb_filters[1], stride_size=None)

        up6 = self.upconv_unit(inputs=up5_fus, nb_filter=nb_filters[0], concatenate_layer=conv_init_exp)
        up6_fus = self.RRconv_unit(inputs=up6, nb_filter=nb_filters[0], stride_size=None)

        ##========== Output map
        outmap = Conv1D(3, 1, kernel_initializer='he_uniform', name='pred_label')(up6_fus)#, kernel_regularizer=l2(1e-3)
        outmask = Conv1D(2, 1, kernel_initializer='he_uniform', name='pred_mask')(up6_fus)
        outmap_Act = Activation(self.out_activation)(outmap)
        outmask_Act = Activation(self.out_activation)(outmask)
        model = Model(inputs=inputs, outputs=[outmap_Act, outmask_Act])

        # compile
        if pretrained_weights==None:
            return model             

        else:
            model.load_weights(pretrained_weights)    
            return model

    def build_attunet(self, pretrained_weights=None,input_size=None, nb_filters=None):
        if nb_filters == None:
            nb_filters = self.nb_filters
        if input_size == None:
            input_size = self.input_size
        inputs = Input(input_size)
        conv_init_exp = self.conv_unit(inputs=inputs, nb_filter=nb_filters[0], stride_size=None)   
        # (None, 1, nb_filter[0], 2001)
        #========== Encoder
        down1 = self.conv_unit(inputs=conv_init_exp, nb_filter=nb_filters[0], stride_size=self.stride_size) # downsample
        down1_exp = self.conv_unit(inputs=down1, nb_filter=nb_filters[1], stride_size=None)
        # (None, 501, nb_filter[0])
        down2 = self.conv_unit(inputs=down1_exp, nb_filter=nb_filters[1], stride_size=self.stride_size)
        down2_exp = self.conv_unit(inputs=down2, nb_filter=nb_filters[2], stride_size=None)
        # (None, 126, nb_filter[1])
        down3 = self.conv_unit(inputs=down2_exp, nb_filter=nb_filters[2], stride_size=self.stride_size)
        down3_exp = self.conv_unit(inputs=down3, nb_filter=nb_filters[3], stride_size=None)
        # (None, 32, nb_filter[2])

        #========== Decoder
        up4 = self.upconv_unit(inputs=down3_exp, nb_filter=nb_filters[2], concatenate_layer=down2_exp, apply_attention=True)
        up4_fus = self.conv_unit(inputs=up4, nb_filter=nb_filters[2], stride_size=None)

        up5 = self.upconv_unit(inputs=up4_fus, nb_filter=nb_filters[1], concatenate_layer=down1_exp, apply_attention=True)
        up5_fus = self.conv_unit(inputs=up5, nb_filter=nb_filters[1], stride_size=None)

        up6 = self.upconv_unit(inputs=up5_fus, nb_filter=nb_filters[0], concatenate_layer=conv_init_exp, apply_attention=True)
        up6_fus = self.conv_unit(inputs=up6, nb_filter=nb_filters[0], stride_size=None)

        ##========== Output map
        outmap = Conv1D(3, 1, kernel_initializer='he_uniform', name='pred_label')(up6_fus)#, kernel_regularizer=l2(1e-3)
        outmask = Conv1D(2, 1, kernel_initializer='he_uniform', name='pred_mask')(up6_fus)
        outmap_Act = Activation(self.out_activation)(outmap)
        outmask_Act = Activation(self.out_activation)(outmask)
        model = Model(inputs=inputs, outputs=[outmap_Act, outmask_Act])

        # compile
        if pretrained_weights==None:
            return model             

        else:
            model.load_weights(pretrained_weights)    
            return model

    def build_attR2unet(self, pretrained_weights=None, 
                input_size=None, nb_filters=None):
        if nb_filters == None:
            nb_filters = self.nb_filters
        if input_size == None:
            input_size = self.input_size
        inputs = Input(input_size)
        conv_init_exp = self.RRconv_unit(inputs=inputs, 
            nb_filter=nb_filters[0], stride_size=None)   
        # (None, 1, nb_filter[0], 2001)
        #========== Encoder
        down1 = self.RRconv_unit(inputs=conv_init_exp, 
            nb_filter=nb_filters[0], stride_size=self.stride_size) # downsample
        down1_exp = self.RRconv_unit(inputs=down1, 
            nb_filter=nb_filters[1], stride_size=None)
        # (None, 501, nb_filter[0])
        down2 = self.RRconv_unit(inputs=down1_exp, 
            nb_filter=nb_filters[1], stride_size=self.stride_size)
        down2_exp = self.RRconv_unit(inputs=down2, 
            nb_filter=nb_filters[2], stride_size=None)
        # (None, 126, nb_filter[1])
        down3 = self.RRconv_unit(inputs=down2_exp, 
            nb_filter=nb_filters[2], stride_size=self.stride_size)
        down3_exp = self.RRconv_unit(inputs=down3, 
            nb_filter=nb_filters[3], stride_size=None)
        # (None, 32, nb_filter[2])

        #========== Decoder
        up4 = self.upconv_unit(inputs=down3_exp, nb_filter=nb_filters[2], 
            concatenate_layer=down2_exp, apply_attention=True)
        up4_fus = self.RRconv_unit(inputs=up4, nb_filter=nb_filters[2], 
            stride_size=None)

        up5 = self.upconv_unit(inputs=up4_fus, nb_filter=nb_filters[1], 
            concatenate_layer=down1_exp, apply_attention=True)
        up5_fus = self.RRconv_unit(inputs=up5, nb_filter=nb_filters[1], 
            stride_size=None)

        up6 = self.upconv_unit(inputs=up5_fus, nb_filter=nb_filters[0], 
            concatenate_layer=conv_init_exp, apply_attention=True)
        up6_fus = self.RRconv_unit(inputs=up6, nb_filter=nb_filters[0], 
            stride_size=None)

        ##========== Output map
        outmap = Conv1D(3, 1, kernel_initializer='he_uniform',
            name='pred_label')(up6_fus)#, kernel_regularizer=l2(1e-3)
        outmask = Conv1D(2, 1, kernel_initializer='he_uniform', 
            name='pred_mask')(up6_fus)
        outmap_Act = Activation(self.out_activation)(outmap)
        outmask_Act = Activation(self.out_activation)(outmask)
        model = Model(inputs=inputs, outputs=[outmap_Act, outmask_Act])

        # compile
        if pretrained_weights==None:
            return model             

        else:
            model.load_weights(pretrained_weights)    
            return model

if __name__ == '__main__':
    pass
