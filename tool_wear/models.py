from keras.models import Model
from keras.layers.convolutional import Conv1D,MaxPooling1D
from keras.layers.merge import add
from keras.layers import Dropout,Dense,Activation,Flatten,BatchNormalization,Input
from keras.optimizers import Adam
from tool_wear.layers_utils import SpatialPyramidPooling

def preprocess_block(tensor_input,filter_number_list,kernel_size=5,pooling_size=2,dropout_rate=0.5):
    k1, k2 = filter_number_list

    out = Conv1D(k1, 1, padding='same',use_bias=False,kernel_initializer="he_uniform")(tensor_input)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dropout(dropout_rate)(out)
    out = Conv1D(k2, kernel_size, strides=1, padding='same',kernel_initializer="he_uniform")(out)

    pooling = MaxPooling1D(pooling_size, strides=1, padding='same')(tensor_input)

    # out = merge([out,pooling],mode='sum')
    out = add([out, pooling])
    return out

def repeated_block(x,filter_number_list,kernel_size=3,pooling_size=2,dropout_rate=0.5,is_first_layer_of_block=False):

    k1,k2 = filter_number_list
    print(k1,k2)

    out = BatchNormalization()(x)
    out = Activation('relu')(out)
    out = Conv1D(k1,kernel_size,padding='same',use_bias=False,kernel_initializer="he_uniform")(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dropout(dropout_rate)(out)
    out = Conv1D(k2,kernel_size,strides=1,padding='same',kernel_initializer="he_uniform")(out)
    if is_first_layer_of_block:
        # add conv here
        pooling = Conv1D(k2, kernel_size, strides=1, padding='same')(x)
    else:
        pooling = MaxPooling1D(pooling_size, strides=1, padding='same')(x)
        pass

    out = add([out, pooling])
    return out

def build_resnet_with_roi_pooling(input_dim,output_dim,block_number=20,dropout_rate=0.5):

    signal_input = Input(shape=(None,input_dim))

    out = Conv1D(128,3,strides=1,use_bias=False,kernel_initializer="he_uniform")(signal_input)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = preprocess_block(out, (64, 128), dropout_rate=dropout_rate)

    base_filter = 4
    block_part_num = 5

    total_times = block_number // block_part_num

    for cur_layer_num in range(block_number):
        is_first_layer = False
        if cur_layer_num % block_part_num == 0:
            is_first_layer = True
        filter_times = total_times - cur_layer_num // block_part_num
        filter = (base_filter * (2 ** (filter_times)), base_filter * (2 ** (filter_times)))
        out = repeated_block(out, filter, dropout_rate=dropout_rate,is_first_layer_of_block=is_first_layer)

    out = SpatialPyramidPooling([1,2,4])(out)
    # out = Flatten()(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dense(output_dim)(out)

    model = Model(inputs=[signal_input], outputs=[out])
    adam = Adam(lr=0.005)

    model.compile(loss='logcosh', optimizer=adam, metrics=['mse', 'mae'])
    return model
