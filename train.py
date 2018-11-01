import math
import time
import json
import pandas as pd
from utils import splitData
from keras.callbacks import Callback
from keras.utils.vis_utils import plot_model

class LossHistory(Callback):
    def __init__(self, modelObj, filename, N= 1):
        self.modelObj = modelObj
        self.filename = filename
        self.losses = []
        self.N = N
        self.epoch_time_start = 0.0

    def on_train_begin(self,logs={}):
        self.epoch_time_start = time.time()
        # Load Existing CSV is there
        try:
            self.losses = pd.read_csv(self.filename+'_logs.csv').to_dict('records')
        except:
            pass
        # Save Model Config
        data = {}
        data["latent_dim"] = self.modelObj.latent_dim
        data["embedding_size"] = self.modelObj.embedding_size
        data["description"] = self.modelObj.description
        data["attention"] = self.modelObj.attention
        data["lr"] = self.modelObj.lr
        data["dropout"] = self.modelObj.dropout
        data["max_encoder_seq_length"] = self.modelObj.max_encoder_seq_length
        data["max_decoder_seq_length"] = self.modelObj.max_decoder_seq_length
        data["num_encoder_tokens"] = self.modelObj.num_encoder_tokens
        data["num_decoder_tokens"] = self.modelObj.num_decoder_tokens
        data["name"] = self.modelObj.name

        with open(self.filename + '_config.json', 'w') as outfile:
           json.dump(data, outfile, sort_keys = True, indent = 4,
                     ensure_ascii = False)
        print "Saving model config."

    def on_train_end(self,logs={}):
        # Probably print graph and save logs?
        # print self.losses
        pd.DataFrame(self.losses).to_csv(self.filename + '_logs.csv')
        print "Saving all logs."

    def on_epoch_end(self, epochs, logs={}):
        cur_time = time.time()
        logs['eps'] = epochs+1
        logs['ts'] = cur_time - self.epoch_time_start
        self.epoch_time_start = cur_time
        self.losses.append(logs.copy())
        # Save model per N epochs
        if epochs % self.N == 0:
            self.modelObj.model.save(self.filename + '.h5')
            pd.DataFrame(self.losses).to_csv(self.filename + '_logs.csv')
            print "Saving model at", epochs+1


def train(filename, modelObject, df, data_info, gen, batch_size = 128, eps_start = 0, eps_stop = 100, encoding='paddNL'):
    # Split data to train and test
    df_train, df_test = splitData(df)

    # Number of batches of each generator
    train_batch_steps = int(math.ceil(float(df_train.shape[0])/batch_size))
    test_batch_steps = int(math.ceil(float(df_test.shape[0])/batch_size))
    
    # Save Model Flow Image
    showModel(modelObject, filename)

    # History Object for logging and saving
    history = LossHistory(modelObject,filename,1)

    # Run training instance of model
    modelObject.model.fit_generator( gen(df_train, data_info, batch_size, encoding),
            steps_per_epoch = train_batch_steps, 
            initial_epoch = eps_start,
            epochs= eps_stop,
            validation_data = gen(df_test, data_info, batch_size, encoding),
            validation_steps = test_batch_steps,
            callbacks = [history]) 

    # Save model
    modelObject.model.save(filename+'_last.h5')
    print "SAVED!"

def showModel(modelObject, filePath = 'model.png'):
    plot_model(modelObject.model, to_file=filePath + '_model.png', show_shapes=True)
    # plt.rcParams["figure.figsize"] = [20,15]
    # img = mpimg.imread(filePath + '_model.png')
    # imgplot = plt.imshow(img)
    # plt.show()