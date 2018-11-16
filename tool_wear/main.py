from data import FixedIntervalDataset
from keras.callbacks import TensorBoard,ModelCheckpoint
from tool_wear.models import build_resnet_with_roi_pooling
from keras.models import load_model
import os.path
import numpy as np

fixedIntervalData = FixedIntervalDataset()

x = fixedIntervalData.get_all_loc_x_sample_data()
y = fixedIntervalData.get_all_loc_y_sample_data()

# shuffle the dataset
import random
# set random seeds so that we can get the same random data!
SEED = 12347
random.seed(SEED)
index = [i for i in range(len(y))]
random.shuffle(index)
train_y = y[index]
train_x = x[index]

TRAIN_END_INDEX = len(y) // 5 * 4

# ---- function to generate data recursively
def generate_train_signals_batch():
    while True:
        for i in range(TRAIN_END_INDEX):
            yield (np.array([train_x[i]]),np.array([train_y[i]]))

def generate_val_signal_batch():
    while True:
        for i in range(TRAIN_END_INDEX,len(y)):
            yield (np.array([train_x[i]]),np.array([train_y[i]]))

def generate_signal_batch():
    while True:
        for i in range(len(y)):
            yield (np.array([train_x[i]]),np.array([train_y[i]]))

print("""
---------------------------------------
SAMPLE x shape :%s
SAMPLE y shape :%s
---------------------------------------
"""%(x.shape,y.shape))

# -------------CONFIGURATION------------------
LOG_DIR = "KERAS_ROI_POOLING_LOG/"
PREDICT = False
DROPOUT_RATE = 0.1
# -------------END----------------------------

for DEPTH in [20,15,10]:
    TRAIN_NAME = "change_logcosh_resnet_SPP_depth_%s_dropout_%s" %(DEPTH,DROPOUT_RATE)
    MODEL_NAME = "%s.kerasmodel"%(TRAIN_NAME)
    MODEL_WEIGHT_NAME = "%s.kerasweight"%(TRAIN_NAME)
    MODEL_CHECK_PT = "%s.kerascheckpts"%(TRAIN_NAME)

    # for i in range(len(y)):
    #     print("x SHAPE :",i,x[i].shape)

    print("""
---------------------------------------
SAMPLE TRAIN x shape :%s
SAMPLE TRAIN y shape :%s
---------------------------------------
    """ % (train_x.shape, train_y.shape))

    if not PREDICT:
        print("In [TRAIN] mode")
        # CALLBACK
        tb_cb = TensorBoard(log_dir=LOG_DIR+TRAIN_NAME)
        ckp_cb = ModelCheckpoint(MODEL_CHECK_PT,monitor='val_loss',save_weights_only=True,verbose=1,save_best_only=True,period=5)

        model = build_resnet_with_roi_pooling(7,3,block_number=DEPTH,dropout_rate=DROPOUT_RATE)
        print(model.summary())
        print("MODEL has been built ...")
        if os.path.exists(MODEL_CHECK_PT):
            model.load_weights(MODEL_CHECK_PT)
            print("load checkpoint successfully")
        else:
            print("No checkpoints found !")
        print("Start to train the model")
        model.fit_generator(generate_train_signals_batch(),
                            steps_per_epoch=TRAIN_END_INDEX,
                            epochs=1000,
                            callbacks=[tb_cb,ckp_cb],
                            validation_data=generate_val_signal_batch(),
                            validation_steps=len(y)-TRAIN_END_INDEX,
                            use_multiprocessing=True
                            )
        # model.fit(np.array([train_x[0]]),np.array([train_y[0]]),batch_size=1,epochs=1000,validation_split=0.2)
        model.model.save(MODEL_NAME)
        model.save_weights(MODEL_WEIGHT_NAME)
    else:
        print("In [EVALUATE] mode")
        import matplotlib.pyplot as plt
        import matplotlib
        import matplotlib.mlab as mlab
        # load check point instead
        model = build_resnet_with_roi_pooling(7, 3, block_number=DEPTH, dropout_rate=DROPOUT_RATE)
        if os.path.exists(MODEL_CHECK_PT):
            model.load_weights(MODEL_CHECK_PT)
            print("load checkpoint successfully")
        else:
            print("No checkpoints found and trying to load model directly")
            model = load_model(MODEL_NAME)
        y_pred = model.predict_generator(generate_signal_batch(),steps=len(y),verbose=1)
        for i in range(3):
            fig = plt.figure()
            plt.plot(y[:,i],label="real")
            plt.plot(y_pred[:,i],label="predicted")
            plt.legend()
            fig.show()
