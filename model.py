from keras.models import Model, load_model
from keras.layers import Input, LSTM, GRU, Dense, Bidirectional, Dropout, Concatenate, Embedding, dot, Activation, concatenate, TimeDistributed
from keras import optimizers
from metrics import *
import json

class GRU_MODEL():
    def __init__(self, data_info, latent_dim=64, embedding_size = 40, dropout=0.1, lr=0.001, attention = False):
        self.description = 'GRU-1-layer-Encoder GRU-1-layer-Decoder'
        self.attention = attention
        self.max_encoder_seq_length = data_info["max_encoder_seq_length"]
        self.max_decoder_seq_length = data_info["max_decoder_seq_length"]
        self.num_encoder_tokens = data_info["num_encoder_tokens"]
        self.num_decoder_tokens = data_info["num_decoder_tokens"]

        self.latent_dim = latent_dim
        self.dropout = dropout
        self.lr = lr
        self.embedding_size = embedding_size
        self.model = None
        self.name = 'GRU'

    def buildModel(self):
        # ENCODER
        encoder_inputs = Input(shape=(None, ))
        encoder_embedding = Embedding(self.num_encoder_tokens, self.embedding_size, mask_zero = True)(encoder_inputs)
        # layer 1
        encoder = GRU(self.latent_dim, return_sequences=True, return_state=True, dropout=self.dropout)
        encoder_outputs, enc_state_h = encoder(encoder_embedding)
        # encoder states
        encoder_states = [enc_state_h]
        # DECODER
        decoder_inputs = Input(shape=(None, ))
        decoder_embedding = Embedding(self.num_decoder_tokens, self.embedding_size, mask_zero = True)(decoder_inputs)
        # layer 1
        decoder_gru = GRU(self.latent_dim, return_sequences=True, return_state=True, dropout=self.dropout)
        decoder_outputs, _ = decoder_gru(decoder_embedding, initial_state=encoder_states)
        # Attention
        if self.attention:
            attention = dot([decoder_outputs, encoder_outputs], axes=[2, 2])
            attention = Activation('softmax', name='attention')(attention)
            context = dot([attention, encoder_outputs], axes=[2,1])
            decoder_combined_context = concatenate([context, decoder_outputs])
            decoder_outputs = TimeDistributed(Dense(self.embedding_size, activation="tanh"))(decoder_combined_context)
        # dense layer for categorical one-hot output
        outputs = TimeDistributed(Dense(self.num_decoder_tokens, activation="softmax"))(decoder_outputs)
        # Model Compilation
        adam = optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model = Model([encoder_inputs, decoder_inputs], outputs)
        model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['acc', precision, recall, f1])
        self.model = model

    def loadModel(self, modelPath):
        self.loadConfig(modelPath)
        self.model = load_model(modelPath + '.h5',  custom_objects={'precision': precision, 'recall': recall, 'f1':f1})
        print "Loaded:", modelPath

    def loadConfig(self, filename):
        with open(filename + '_config.json', 'r') as outfile:
           data = json.load(outfile)
        self.latent_dim = data["latent_dim"]
        self.embedding_size = data["embedding_size"]
        self.description = data["description"]
        self.attention = data["attention"]
        self.lr = data["lr"]
        self.dropout = data["dropout"]
        self.max_encoder_seq_length = data["max_encoder_seq_length"]
        self.max_decoder_seq_length = data["max_decoder_seq_length"]
        self.num_encoder_tokens = data["num_encoder_tokens"]
        self.num_decoder_tokens = data["num_decoder_tokens"]
        self.name = data["name"]

    def getDecoder(self, filename):
        model = load_model(filename,  custom_objects={'precision': precision, 'recall': recall, 'f1':f1})
        # ENCODER
        encoder_inputs = model.input[0] #input_1
        encoder_outputs, state_h_enc = model.layers[4].output
        encoder_states = [state_h_enc]
        # SET ENCODER MODEL
        encoder_model = Model(encoder_inputs,[encoder_outputs] + encoder_states)
        # DECODER
        decoder_inputs = model.input[1] #input_2
        decoder_embedding = model.layers[3](decoder_inputs)
        # layer1
        dec_input_h = Input(shape=(self.latent_dim,))
        dec_state_input = [dec_input_h]
        decoder_gru = model.layers[5]
        decoder1_outputs, dec_state_h = decoder_gru(decoder_embedding, initial_state=dec_state_input)
        # States and Input
        dec_state_input = [dec_input_h]
        decoder_states = [dec_state_h]
        encoder1_outputs = Input(shape=(None, self.latent_dim))
        # ATTENTION
        dot_layer1 = model.layers[6]
        attn_inp = dot_layer1([decoder1_outputs, encoder1_outputs])
        act_layer = model.layers[7]
        attn_out = act_layer(attn_inp)
        dot_layer2 = model.layers[8]
        cntxt1 = dot_layer2([attn_out, encoder1_outputs])
        concat_layer = model.layers[9]
        cntxt2 = concat_layer([cntxt1, decoder1_outputs])
        tdt_layer = model.layers[10]
        cntxt3 = tdt_layer(cntxt2)
        # TIME DISTRIBUTED DENSE
        tds_layer = model.layers[11]
        output = tds_layer(cntxt3)
        # SET DECODER MODEL
        decoder_model = Model([encoder1_outputs, decoder_inputs] + dec_state_input, [output] + decoder_states)
        model = None
        return encoder_model, decoder_model

class BiGRU_MODEL():
    def __init__(self, data_info, latent_dim=64, embedding_size = 40, dropout=0.1, lr=0.001, attention = False):
        self.description = "BidirectionGRU-1-layer-Encoder GRU-1-layer-Decoder"
        self.attention = attention

        self.max_encoder_seq_length = data_info["max_encoder_seq_length"]
        self.max_decoder_seq_length = data_info["max_decoder_seq_length"]
        self.num_encoder_tokens = data_info["num_encoder_tokens"]
        self.num_decoder_tokens = data_info["num_decoder_tokens"]

        self.latent_dim = latent_dim
        self.dropout = dropout
        self.lr = lr
        self.embedding_size = embedding_size
        self.model = None
        self.name = 'BiGRU'

    def buildModel(self): 
        # ENCODER
        encoder_inputs = Input(shape=(None, )) 
        encoder_embedding = Embedding(self.num_encoder_tokens, self.embedding_size, mask_zero = True)(encoder_inputs)
        # layer 1
        encoder = GRU(self.latent_dim, return_sequences=True, return_state=True, dropout=self.dropout)
        bi_encoder = Bidirectional(encoder, merge_mode='concat')
        encoder_outputs, enc_fwd_h, enc_back_h = bi_encoder(encoder_embedding)
        enc_state_h = Concatenate()([enc_fwd_h, enc_back_h])
        # encoder states
        encoder_states = [enc_state_h]
        # DECODER
        decoder_inputs = Input(shape=(None, ))
        decoder_embedding = Embedding(self.num_decoder_tokens, self.embedding_size,  mask_zero = True)(decoder_inputs)
        # layer 1
        decoder_gru = GRU(2 * self.latent_dim, return_sequences=True, return_state=True, dropout=self.dropout)
        decoder_outputs, _ = decoder_gru(decoder_embedding, initial_state=encoder_states)
        # Attention
        if self.attention:
            attention = dot([decoder_outputs, encoder_outputs], axes=[2, 2])
            attention = Activation('softmax', name='attention')(attention)
            context = dot([attention, encoder_outputs], axes=[2,1])
            decoder_combined_context = concatenate([context, decoder_outputs])
            decoder_outputs = TimeDistributed(Dense(self.embedding_size, activation="tanh"))(decoder_combined_context)
        # dense layer for categorical one-hot output
        outputs = TimeDistributed(Dense(self.num_decoder_tokens, activation="softmax"))(decoder_outputs)
        # Model Compilation
        adam = optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model = Model([encoder_inputs, decoder_inputs], outputs)
        model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['acc', precision, recall, f1])
        self.model = model

    def loadModel(self, modelPath):
        self.loadConfig(modelPath)
        self.model = load_model(modelPath + '.h5',  custom_objects={'precision': precision, 'recall': recall, 'f1':f1})
        print "Loaded:", modelPath

    def loadConfig(self, filename):
        with open(filename + '_config.json', 'r') as outfile:
           data = json.load(outfile)
        self.latent_dim = data["latent_dim"]
        self.embedding_size = data["embedding_size"]
        self.description = data["description"]
        self.attention = data["attention"]
        self.lr = data["lr"]
        self.dropout = data["dropout"]
        self.max_encoder_seq_length = data["max_encoder_seq_length"]
        self.max_decoder_seq_length = data["max_decoder_seq_length"]
        self.num_encoder_tokens = data["num_encoder_tokens"]
        self.num_decoder_tokens = data["num_decoder_tokens"]
        self.name = data["name"]

    def getDecoder(self, filename):
        model = load_model(filename,  custom_objects={'precision': precision, 'recall': recall, 'f1':f1})
        # ENCODER
        encoder_inputs = model.input[0] #input_1
        state_h_enc = model.layers[5].output
        encoder_states = [state_h_enc]
        encoder_outputs,_,_ = model.layers[3].output
        # SET ENCODER MODEL
        encoder_model = Model(encoder_inputs,[encoder_outputs] + encoder_states)
        # DECODER
        decoder_inputs = model.input[1] #input_2
        decoder_embedding = model.layers[4](decoder_inputs)
        # layer1
        dec_input_h = Input(shape=(2*self.latent_dim,))
        dec_state_input = [dec_input_h]
        decoder_gru = model.layers[6]
        decoder1_outputs, dec_state_h = decoder_gru(decoder_embedding, initial_state=dec_state_input)
        # States and Input
        dec_state_input = [dec_input_h]
        decoder_states = [dec_state_h]
        encoder1_outputs = Input(shape=(None, 2*self.latent_dim))
        # ATTENTION
        dot_layer1 = model.layers[7]
        attn_inp = dot_layer1([decoder1_outputs, encoder1_outputs])
        act_layer = model.layers[8]
        attn_out = act_layer(attn_inp)
        dot_layer2 = model.layers[9]
        cntxt1 = dot_layer2([attn_out, encoder1_outputs])
        concat_layer = model.layers[10]
        cntxt2 = concat_layer([cntxt1, decoder1_outputs])
        tdt_layer = model.layers[11]
        cntxt3 = tdt_layer(cntxt2)
        # TIME DISTRIBUTED DENSE
        tds_layer = model.layers[12]
        output = tds_layer(cntxt3)
        # SET DECODER MODEL
        decoder_model = Model([encoder1_outputs, decoder_inputs] + dec_state_input, [output] + decoder_states)
        model = None
        return encoder_model, decoder_model

class BiGRU2_MODEL():
    def __init__(self, data_info, latent_dim=64, embedding_size = 40, dropout=0.1, lr=0.001, attention = False):
        self.description = "BidirectionGRU-2-layer-Encoder GRU-2-layer-Decoder"
        self.attention = attention

        self.max_encoder_seq_length = data_info["max_encoder_seq_length"]
        self.max_decoder_seq_length = data_info["max_decoder_seq_length"]
        self.num_encoder_tokens = data_info["num_encoder_tokens"]
        self.num_decoder_tokens = data_info["num_decoder_tokens"]

        self.latent_dim = latent_dim
        self.dropout = dropout
        self.lr = lr
        self.embedding_size = embedding_size
        self.model = None
        self.name = 'BiGRU2'

    def buildModel(self):
        # ENCODER
        encoder_inputs = Input(shape=(None, )) 
        encoder_embedding = Embedding(self.num_encoder_tokens, self.embedding_size, mask_zero = True)(encoder_inputs)
        # layer 1
        encoder_gru1 = GRU(self.latent_dim, return_sequences= True, return_state=True, dropout=self.dropout)    
        bi_encoder1 = Bidirectional(encoder_gru1, merge_mode='concat')
        encoder1_outputs, enc1_fwd_h, enc1_back_h = bi_encoder1(encoder_embedding)
        enc1_state_h = Concatenate()([enc1_fwd_h, enc1_back_h])
        encoder1_states = [enc1_state_h]
        # layer 2
        encoder_gru2 = GRU(self.latent_dim, return_sequences= True, return_state=True, dropout=self.dropout)
        bi_encoder2 = Bidirectional(encoder_gru2, merge_mode='concat')
        encoder2_outputs, enc2_fwd_h, enc2_back_h = bi_encoder2(encoder1_outputs)
        enc2_state_h = Concatenate()([enc2_fwd_h, enc2_back_h])
        encoder2_states = [enc2_state_h]
        # encoder states
        encoder_states = [enc1_state_h, enc2_state_h] 
        # DECODER
        decoder_inputs = Input(shape=(None, ))
        decoder_embedding = Embedding(self.num_decoder_tokens, self.embedding_size,  mask_zero = True)(decoder_inputs)
        # layer 1
        decoder_gru1 = GRU(2 * self.latent_dim, return_sequences=True, return_state=True, dropout=self.dropout)
        decoder1_outputs, _ = decoder_gru1(decoder_embedding, initial_state=encoder1_states)
        # layer 2
        decoder_gru2 = GRU(2 * self.latent_dim, return_sequences=True, return_state=True, dropout=self.dropout)
        decoder2_outputs, _ = decoder_gru2(decoder1_outputs, initial_state=encoder2_states)
        # Attention
        if self.attention:
            attention = dot([decoder2_outputs, encoder2_outputs], axes=[2, 2])
            attention = Activation('softmax', name='attention')(attention)
            context = dot([attention, encoder2_outputs], axes=[2,1])
            decoder_combined_context = concatenate([context, decoder2_outputs])
            decoder2_outputs = TimeDistributed(Dense(self.embedding_size, activation="tanh"))(decoder_combined_context)
        # dense layer for categorical one-hot output
        outputs = TimeDistributed(Dense(self.num_decoder_tokens, activation="softmax"))(decoder2_outputs)
        # Model Compilation
        adam = optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model = Model([encoder_inputs, decoder_inputs], outputs)
        model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['acc', precision, recall, f1])
        self.model = model

    def loadModel(self, modelPath):
        self.loadConfig(modelPath)
        self.model = load_model(modelPath + '.h5',  custom_objects={'precision': precision, 'recall': recall, 'f1':f1})
        print "Loaded:", modelPath

    def loadConfig(self, filename):
        with open(filename + '_config.json', 'r') as outfile:
           data = json.load(outfile)
        self.latent_dim = data["latent_dim"]
        self.embedding_size = data["embedding_size"]
        self.description = data["description"]
        self.attention = data["attention"]
        self.lr = data["lr"]
        self.dropout = data["dropout"]
        self.max_encoder_seq_length = data["max_encoder_seq_length"]
        self.max_decoder_seq_length = data["max_decoder_seq_length"]
        self.num_encoder_tokens = data["num_encoder_tokens"]
        self.num_decoder_tokens = data["num_decoder_tokens"]
        self.name = data["name"]

    def getDecoder(self, filename):
        model = load_model(filename,  custom_objects={'precision': precision, 'recall': recall, 'f1':f1})
        # ENCODER
        encoder_inputs = model.input[0] #input_1
        enc1_state_h = model.layers[5].output
        enc2_state_h = model.layers[8].output
        encoder_states = [enc1_state_h, enc2_state_h]
        encoder_outputs,_,_ = model.layers[6].output
        # SET ENCODER MODEL
        encoder_model = Model(encoder_inputs,[encoder_outputs] + encoder_states)
        # DECODER
        decoder_inputs = model.input[1] #input_2
        decoder_embedding = model.layers[4](decoder_inputs)
        # layer1
        dec1_input_h = Input(shape=(2*self.latent_dim,))
        dec1_state_input = [dec1_input_h]
        decoder_gru1 = model.layers[7]
        decoder1_outputs, dec1_state_h = decoder_gru1(decoder_embedding, initial_state=dec1_state_input)
        # layer2
        dec2_input_h = Input(shape=(2*self.latent_dim,))
        dec2_state_input = [dec2_input_h]
        decoder_gru2 = model.layers[9]
        decoder2_outputs, dec2_state_h = decoder_gru2(decoder1_outputs, initial_state=dec2_state_input)
        # States and Input
        dec_state_input = [dec1_input_h, dec2_input_h]
        decoder_states = [dec1_state_h, dec2_state_h]
        encoder2_outputs = Input(shape=(None, 2*self.latent_dim))
        # ATTENTION
        dot_layer1 = model.layers[10]
        attn_inp = dot_layer1([decoder2_outputs, encoder2_outputs])
        act_layer = model.layers[11]
        attn_out = act_layer(attn_inp)
        dot_layer2 = model.layers[12]
        cntxt1 = dot_layer2([attn_out, encoder2_outputs])
        concat_layer = model.layers[13]
        cntxt2 = concat_layer([cntxt1, decoder2_outputs])
        # TIME DISTRIBUTED
        tdt_layer = model.layers[14]
        cntxt3 = tdt_layer(cntxt2)
        tds_layer = model.layers[15]
        output = tds_layer(cntxt3)
        # SET DECODER MODEL
        decoder_model = Model([encoder2_outputs, decoder_inputs] + dec_state_input, [output] + decoder_states)
        model = None
        return encoder_model, decoder_model

class GRU2_MODEL():
    def __init__(self, data_info, latent_dim=64, embedding_size = 40, dropout=0.1, lr=0.001, attention = False):
        self.description = "GRU-2-layer-Encoder GRU-2-layer-Decoder"
        self.attention = attention

        self.max_encoder_seq_length = data_info["max_encoder_seq_length"]
        self.max_decoder_seq_length = data_info["max_decoder_seq_length"]
        self.num_encoder_tokens = data_info["num_encoder_tokens"]
        self.num_decoder_tokens = data_info["num_decoder_tokens"]

        self.latent_dim = latent_dim
        self.dropout = dropout
        self.lr = lr
        self.embedding_size = embedding_size
        self.model = None
        self.name = 'GRU2'

    def buildModel(self):
        # ENCODER
        encoder_inputs = Input(shape=(None, )) 
        encoder_embedding = Embedding(self.num_encoder_tokens, self.embedding_size, mask_zero = True)(encoder_inputs)
        # layer 1
        encoder_gru1 = GRU(self.latent_dim, return_sequences= True, return_state=True, dropout=self.dropout)    
        encoder1_outputs, enc1_state_h = encoder_gru1(encoder_embedding)
        encoder1_states = [enc1_state_h]
        # layer 2
        encoder_gru2 = GRU(self.latent_dim, return_sequences= True, return_state=True, dropout=self.dropout)
        encoder2_outputs, enc2_state_h = encoder_gru2(encoder1_outputs)
        encoder2_states = [enc2_state_h]
        # encoder states
        encoder_states = [enc1_state_h, enc2_state_h]  
        # DECODER
        decoder_inputs = Input(shape=(None, ))
        decoder_embedding = Embedding(self.num_decoder_tokens, self.embedding_size,  mask_zero = True)(decoder_inputs)
        # layer 1
        decoder_gru1 = GRU(self.latent_dim, return_sequences=True, return_state=True, dropout=self.dropout)
        decoder1_outputs, _ = decoder_gru1(decoder_embedding, initial_state=encoder1_states)
        # layer 2
        decoder_gru2 = GRU(self.latent_dim, return_sequences=True, return_state=True, dropout=self.dropout)
        decoder2_outputs, _ = decoder_gru2(decoder1_outputs, initial_state=encoder2_states)
        # Attention
        if self.attention:
            attention = dot([decoder2_outputs, encoder2_outputs], axes=[2, 2])
            attention = Activation('softmax', name='attention')(attention)
            context = dot([attention, encoder2_outputs], axes=[2,1])
            decoder_combined_context = concatenate([context, decoder2_outputs])
            decoder2_outputs = TimeDistributed(Dense(self.embedding_size, activation="tanh"))(decoder_combined_context)
        # dense layer for categorical one-hot output
        outputs = TimeDistributed(Dense(self.num_decoder_tokens, activation="softmax"))(decoder2_outputs)
        # Model Compilation
        adam = optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model = Model([encoder_inputs, decoder_inputs], outputs)
        model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['acc', precision, recall, f1])
        self.model = model

    def loadModel(self, modelPath):
        self.loadConfig(modelPath)
        self.model = load_model(modelPath + '.h5',  custom_objects={'precision': precision, 'recall': recall, 'f1':f1})
        print "Loaded:", modelPath

    def loadConfig(self, filename):
        with open(filename + '_config.json', 'r') as outfile:
           data = json.load(outfile)
        self.latent_dim = data["latent_dim"]
        self.embedding_size = data["embedding_size"]
        self.description = data["description"]
        self.attention = data["attention"]
        self.lr = data["lr"]
        self.dropout = data["dropout"]
        self.max_encoder_seq_length = data["max_encoder_seq_length"]
        self.max_decoder_seq_length = data["max_decoder_seq_length"]
        self.num_encoder_tokens = data["num_encoder_tokens"]
        self.num_decoder_tokens = data["num_decoder_tokens"]
        self.name = data["name"]

    def getDecoder(self, filename):
        model = load_model(filename,  custom_objects={'precision': precision, 'recall': recall, 'f1':f1})
        # ENCODER
        encoder_inputs = model.input[0] #input_1
        _, enc1_state_h = model.layers[4].output
        encoder_outputs, enc2_state_h = model.layers[6].output
        encoder_states = [enc1_state_h, enc2_state_h]
        # SET ENCODER MODEL
        encoder_model = Model(encoder_inputs,[encoder_outputs] + encoder_states)
        # DECODER
        decoder_inputs = model.input[1] #input_2
        decoder_embedding = model.layers[3](decoder_inputs)
        # layer1
        dec1_input_h = Input(shape=(self.latent_dim,))
        dec1_state_input = [dec1_input_h]
        decoder_gru1 = model.layers[5]
        decoder1_outputs, dec1_state_h = decoder_gru1(decoder_embedding, initial_state=dec1_state_input)
        # layer2
        dec2_input_h = Input(shape=(self.latent_dim,))
        dec2_state_input = [dec2_input_h]
        decoder_gru2 = model.layers[7]
        decoder2_outputs, dec2_state_h = decoder_gru2(decoder1_outputs, initial_state=dec2_state_input)
        # States and Input
        dec_state_input = [dec1_input_h, dec2_input_h]
        decoder_states = [dec1_state_h, dec2_state_h]
        encoder2_outputs = Input(shape=(None, self.latent_dim))
        # ATTENTION
        dot_layer1 = model.layers[8]
        attn_inp = dot_layer1([decoder2_outputs, encoder2_outputs])
        act_layer = model.layers[9]
        attn_out = act_layer(attn_inp)
        dot_layer2 = model.layers[10]
        cntxt1 = dot_layer2([attn_out, encoder2_outputs])
        concat_layer = model.layers[11]
        cntxt2 = concat_layer([cntxt1, decoder2_outputs])
        # TIME DISTRIBUTED
        tdt_layer = model.layers[12]
        cntxt3 = tdt_layer(cntxt2)
        tds_layer = model.layers[13]
        output = tds_layer(cntxt3)
        # SET DECODER MODEL
        decoder_model = Model([encoder2_outputs, decoder_inputs] + dec_state_input, [output] + decoder_states)
        model = None
        return encoder_model, decoder_model

class LSTM_MODEL():
    def __init__(self, data_info, latent_dim=64, embedding_size = 40, dropout=0.1, lr=0.001, attention = False):
        self.description = "LSTM-1-layer-Encoder LSTM-1-layer-Decoder"
        self.attention = attention
        
        self.max_encoder_seq_length = data_info["max_encoder_seq_length"]
        self.max_decoder_seq_length = data_info["max_decoder_seq_length"]
        self.num_encoder_tokens = data_info["num_encoder_tokens"]
        self.num_decoder_tokens = data_info["num_decoder_tokens"]
        
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.lr = lr
        self.embedding_size = embedding_size
        self.model = None
        self.name = 'LSTM'
        
    def buildModel(self):
        # ENCODER
        encoder_inputs = Input(shape=(None, ))
        encoder_embedding = Embedding(self.num_encoder_tokens, self.embedding_size, mask_zero = True)(encoder_inputs)
        # layer 1
        encoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True, dropout= self.dropout)
        encoder_outputs, enc_state_h, enc_state_c = encoder_lstm(encoder_embedding)
        # encoder states
        encoder_states = [enc_state_h, enc_state_c]
        # DECODER
        decoder_inputs = Input(shape=(None, ))
        decoder_embedding = Embedding(self.num_decoder_tokens, self.embedding_size,  mask_zero = True)(decoder_inputs)
        # layer 1
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True, dropout= self.dropout)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
        # Attention
        if self.attention:
            attention = dot([decoder_outputs, encoder_outputs], axes=[2, 2])
            attention = Activation('softmax', name='attention')(attention)
            context = dot([attention, encoder_outputs], axes=[2,1])
            decoder_combined_context = concatenate([context, decoder_outputs])
            decoder_outputs = TimeDistributed(Dense(self.embedding_size, activation="tanh"))(decoder_combined_context)
        # dense layer for categorical one-hot output
        outputs = TimeDistributed(Dense(self.num_decoder_tokens, activation="softmax"))(decoder_outputs)
        # Model Compilation
        adam = optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model = Model([encoder_inputs, decoder_inputs], outputs)
        model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['acc', precision, recall, f1])
        self.model = model

    def loadModel(self, modelPath):
        self.loadConfig(modelPath)
        self.model = load_model(modelPath + '.h5',  custom_objects={'precision': precision, 'recall': recall, 'f1':f1})
        print "Loaded:", modelPath

    def loadConfig(self, filename):
        with open(filename + '_config.json', 'r') as outfile:
           data = json.load(outfile)
        self.latent_dim = data["latent_dim"]
        self.embedding_size = data["embedding_size"]
        self.description = data["description"]
        self.attention = data["attention"]
        self.lr = data["lr"]
        self.dropout = data["dropout"]
        self.max_encoder_seq_length = data["max_encoder_seq_length"]
        self.max_decoder_seq_length = data["max_decoder_seq_length"]
        self.num_encoder_tokens = data["num_encoder_tokens"]
        self.num_decoder_tokens = data["num_decoder_tokens"]
        self.name = data["name"]

    def getDecoder(self, filename):
        model = load_model(filename,  custom_objects={'precision': precision, 'recall': recall, 'f1':f1})
        # ENCODER
        encoder_inputs = model.input[0] #input_1
        encoder_outputs, state_h_enc, state_c_enc = model.layers[4].output
        encoder_states = [state_h_enc, state_c_enc]
        # SET ENCODER MODEL
        encoder_model = Model(encoder_inputs,[encoder_outputs] + encoder_states)
        # DECODER
        decoder_inputs = model.input[1] #input_2
        decoder_embedding = model.layers[3](decoder_inputs)
        # layer1
        dec_input_h = Input(shape=(self.latent_dim,))
        dec_input_c = Input(shape=(self.latent_dim,))
        dec_state_input = [dec_input_h, dec_input_c]
        decoder_lstm = model.layers[5]
        decoder1_outputs, dec_state_h = decoder_lstm(decoder_embedding, initial_state=dec_state_input)
        # States and Input
        dec_state_input = [dec_input_h, dec_input_c]
        decoder_states = [dec_state_h, dec_state_c]
        encoder1_outputs = Input(shape=(None, self.latent_dim))
        # ATTENTION
        dot_layer1 = model.layers[6]
        attn_inp = dot_layer1([decoder1_outputs, encoder1_outputs])
        act_layer = model.layers[7]
        attn_out = act_layer(attn_inp)
        dot_layer2 = model.layers[8]
        cntxt1 = dot_layer2([attn_out, encoder1_outputs])
        concat_layer = model.layers[9]
        cntxt2 = concat_layer([cntxt1, decoder1_outputs])
        tdt_layer = model.layers[10]
        cntxt3 = tdt_layer(cntxt2)
        # TIME DISTRIBUTED DENSE
        tds_layer = model.layers[11]
        output = tds_layer(cntxt3)
        # SET DECODER MODEL
        decoder_model = Model([encoder1_outputs, decoder_inputs] + dec_state_input, [output] + decoder_states)
        model = None
        return encoder_model, decoder_model

class BiLSTM_MODEL():
    def __init__(self, data_info, latent_dim=64, embedding_size = 40, dropout=0.1, lr=0.001, attention = False):
        self.description = "BidirectionLSTM-1-layer-Encoder LSTM-1-layer-Decoder"
        self.attention = attention

        self.max_encoder_seq_length = data_info["max_encoder_seq_length"]
        self.max_decoder_seq_length = data_info["max_decoder_seq_length"]
        self.num_encoder_tokens = data_info["num_encoder_tokens"]
        self.num_decoder_tokens = data_info["num_decoder_tokens"]

        self.latent_dim = latent_dim
        self.dropout = dropout
        self.lr = lr
        self.embedding_size = embedding_size
        self.model = None
        self.name = 'BiLSTM'
        
    def buildModel(self):
        # ENCODER
        encoder_inputs = Input(shape=(None, )) 
        encoder_embedding = Embedding(self.num_encoder_tokens, self.embedding_size, mask_zero = True)(encoder_inputs)
        # layer 1
        encoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True, dropout= self.dropout)
        bi_encoder = Bidirectional(encoder_lstm, merge_mode='concat')
        encoder_outputs, enc_fwd_h, enc_fwd_c, enc_back_h, enc_back_c = bi_encoder(encoder_embedding)
        enc_state_h = Concatenate()([enc_fwd_h, enc_back_h])
        enc_state_c = Concatenate()([enc_fwd_c, enc_back_c])
        # encoder states
        encoder_states = [enc_state_h, enc_state_c]
        # DECODER
        decoder_inputs = Input(shape=(None, ))
        decoder_embedding = Embedding(self.num_decoder_tokens, self.embedding_size,  mask_zero = True)(decoder_inputs)
        # layer 1
        decoder_lstm = LSTM(2 * self.latent_dim, return_sequences=True, return_state=True, dropout= self.dropout)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
        # Attention
        if self.attention:
            attention = dot([decoder_outputs, encoder_outputs], axes=[2, 2])
            attention = Activation('softmax', name='attention')(attention)
            context = dot([attention, encoder_outputs], axes=[2,1])
            decoder_combined_context = concatenate([context, decoder_outputs])
            decoder_outputs = TimeDistributed(Dense(self.embedding_size, activation="tanh"))(decoder_combined_context)
        # dense layer for categorical one-hot output
        outputs = TimeDistributed(Dense(self.num_decoder_tokens, activation="softmax"))(decoder_outputs)
        # Model Compilation
        adam = optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model = Model([encoder_inputs, decoder_inputs], outputs)
        model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['acc', precision, recall, f1])
        self.model = model
        
    def loadModel(self, modelPath):
        self.loadConfig(modelPath)
        self.model = load_model(modelPath + '.h5',  custom_objects={'precision': precision, 'recall': recall, 'f1':f1})
        print "Loaded:", modelPath

    def loadConfig(self, filename):
        with open(filename + '_config.json', 'r') as outfile:
           data = json.load(outfile)
        self.latent_dim = data["latent_dim"]
        self.embedding_size = data["embedding_size"]
        self.description = data["description"]
        self.attention = data["attention"]
        self.lr = data["lr"]
        self.dropout = data["dropout"]
        self.max_encoder_seq_length = data["max_encoder_seq_length"]
        self.max_decoder_seq_length = data["max_decoder_seq_length"]
        self.num_encoder_tokens = data["num_encoder_tokens"]
        self.num_decoder_tokens = data["num_decoder_tokens"]
        self.name = data["name"]

    def getDecoder(self, filename):
        model = load_model(filename,  custom_objects={'precision': precision, 'recall': recall, 'f1':f1})
        # ENCODER
        encoder_inputs = model.input[0] #input_1
        enc_state_h = model.layers[5].output
        enc_state_c = model.layers[6].output
        encoder_states = [enc_state_h, enc_state_c]
        encoder_outputs,_,_,_,_ = model.layers[3].output
        # SET ENCODER MODEL
        encoder_model = Model(encoder_inputs,[encoder_outputs] + encoder_states)
        # DECODER
        decoder_inputs = model.input[1] #input_2
        decoder_embedding = model.layers[4](decoder_inputs)
        # layer1
        dec_input_h = Input(shape=(2*self.latent_dim,),name='dec_input_h')
        dec_input_c = Input(shape=(2*self.latent_dim,),name='dec_input_c')
        dec_state_input = [dec_input_h, dec_input_c]
        decoder_lstm = model.layers[7]
        decoder1_outputs, dec_state_h, dec_state_c = decoder_lstm(decoder_embedding, initial_state=dec_state_input)
        # States and Input
        dec_state_input = [dec_input_h, dec_input_c]
        decoder_states = [dec_state_h, dec_state_c]
        encoder1_outputs = Input(shape=(None, 2*self.latent_dim))
        # ATTENTION
        dot_layer1 = model.layers[8]
        attn_inp = dot_layer1([decoder1_outputs, encoder1_outputs])
        act_layer = model.layers[9]
        attn_out = act_layer(attn_inp)
        dot_layer2 = model.layers[10]
        cntxt1 = dot_layer2([attn_out, encoder1_outputs])
        concat_layer = model.layers[11]
        cntxt2 = concat_layer([cntxt1, decoder1_outputs])
        tdt_layer = model.layers[12]
        cntxt3 = tdt_layer(cntxt2)
        # TIME DISTRIBUTED DENSE
        tds_layer = model.layers[13]
        output = tds_layer(cntxt3)
        # SET DECODER MODEL
        decoder_model = Model([encoder1_outputs, decoder_inputs] + dec_state_input, [output] + decoder_states)
        model = None
        return encoder_model, decoder_model

class BiLSTM2_MODEL():
    def __init__(self, data_info, latent_dim=64, embedding_size = 40, dropout=0.1, lr=0.001, attention = False):
        self.description = "BidirectionLSTM-2-layer-Encoder LSTM-2-layer-Decoder"
        self.attention = attention

        self.max_encoder_seq_length = data_info["max_encoder_seq_length"]
        self.max_decoder_seq_length = data_info["max_decoder_seq_length"]
        self.num_encoder_tokens = data_info["num_encoder_tokens"]
        self.num_decoder_tokens = data_info["num_decoder_tokens"]

        self.latent_dim = latent_dim
        self.dropout = dropout
        self.lr = lr
        self.embedding_size = embedding_size
        self.model = None
        self.name = 'BiLSTM2'

    def buildModel(self):
        # ENCODER
        encoder_inputs = Input(shape=(None, )) 
        encoder_embedding = Embedding(self.num_encoder_tokens, self.embedding_size, mask_zero = True)(encoder_inputs)
        # layer 1
        encoder_lstm1 = LSTM(self.latent_dim, return_sequences= True, return_state=True, dropout=self.dropout)    
        bi_encoder1 = Bidirectional(encoder_lstm1, merge_mode='concat')
        encoder1_outputs, enc1_fwd_h, enc1_fwd_c, enc1_back_h, enc1_back_c = bi_encoder1(encoder_embedding)
        enc1_state_h = Concatenate()([enc1_fwd_h, enc1_back_h])
        enc1_state_c = Concatenate()([enc1_fwd_c, enc1_back_c])
        encoder1_states = [enc1_state_h, enc1_state_c]
        # layer 2
        encoder_lstm2 = LSTM(self.latent_dim, return_sequences= True, return_state=True, dropout=self.dropout)
        bi_encoder2 = Bidirectional(encoder_lstm2, merge_mode='concat')
        encoder2_outputs, enc2_fwd_h, enc2_fwd_c, enc2_back_h, enc2_back_c = bi_encoder2(encoder1_outputs)
        enc2_state_h = Concatenate()([enc2_fwd_h, enc2_back_h])
        enc2_state_c = Concatenate()([enc2_fwd_c, enc2_back_c])
        encoder2_states = [enc2_state_h, enc2_state_c]
        # encoder states
        encoder_states = [enc1_state_h, enc1_state_c, enc2_state_h, enc2_state_c]         
        # DECODER
        decoder_inputs = Input(shape=(None, ))
        decoder_embedding = Embedding(self.num_decoder_tokens, self.embedding_size,  mask_zero = True)(decoder_inputs)
        # layer 1
        decoder_lstm1 = LSTM(2 * self.latent_dim, return_sequences=True, return_state=True, dropout=self.dropout)
        decoder1_outputs, _, _ = decoder_lstm1(decoder_embedding, initial_state=encoder1_states)
        # layer 2
        decoder_lstm2 = LSTM(2 * self.latent_dim, return_sequences=True, return_state=True, dropout=self.dropout)
        decoder2_outputs, _, _ = decoder_lstm2(decoder1_outputs, initial_state=encoder2_states)
        # Attention
        if self.attention:
            attention = dot([decoder2_outputs, encoder2_outputs], axes=[2, 2])
            attention = Activation('softmax', name='attention')(attention)
            context = dot([attention, encoder2_outputs], axes=[2,1])
            decoder_combined_context = concatenate([context, decoder2_outputs])
            decoder2_outputs = TimeDistributed(Dense(self.embedding_size, activation="tanh"))(decoder_combined_context)
        # dense layer for categorical one-hot output
        outputs = TimeDistributed(Dense(self.num_decoder_tokens, activation="softmax"))(decoder2_outputs)
        # Model Compilation
        adam = optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model = Model([encoder_inputs, decoder_inputs], outputs)
        model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['acc', precision, recall, f1])
        self.model = model

    def loadModel(self, modelPath):
        self.loadConfig(modelPath)
        self.model = load_model(modelPath + '.h5',  custom_objects={'precision': precision, 'recall': recall, 'f1':f1})
        print "Loaded:", modelPath

    def loadConfig(self, filename):
        with open(filename + '_config.json', 'r') as outfile:
           data = json.load(outfile)
        self.latent_dim = data["latent_dim"]
        self.embedding_size = data["embedding_size"]
        self.description = data["description"]
        self.attention = data["attention"]
        self.lr = data["lr"]
        self.dropout = data["dropout"]
        self.max_encoder_seq_length = data["max_encoder_seq_length"]
        self.max_decoder_seq_length = data["max_decoder_seq_length"]
        self.num_encoder_tokens = data["num_encoder_tokens"]
        self.num_decoder_tokens = data["num_decoder_tokens"]
        self.name = data["name"]

    def getDecoder(self, filename):
        model = load_model(filename,  custom_objects={'precision': precision, 'recall': recall, 'f1':f1})
        # ENCODER
        encoder_inputs = model.input[0] #input_1
        enc1_state_h = model.layers[5].output
        enc1_state_c = model.layers[6].output
        enc2_state_h = model.layers[9].output
        enc2_state_c = model.layers[10].output
        encoder_states = [enc1_state_h, enc1_state_c, enc2_state_h, enc2_state_c]
        encoder_outputs,_,_,_,_ = model.layers[7].output
        # SET ENCODER MODEL
        encoder_model = Model(encoder_inputs,[encoder_outputs] + encoder_states)
        # DECODER
        decoder_inputs = model.input[1] #input_2
        decoder_embedding = model.layers[4](decoder_inputs)
        # layer1
        dec1_input_h = Input(shape=(2*self.latent_dim,))
        dec1_input_c = Input(shape=(2*self.latent_dim,))
        dec1_state_input = [dec1_input_h, dec1_input_c]
        decoder_lstm1 = model.layers[8]
        decoder1_outputs, dec1_state_h, dec1_state_c = decoder_lstm1(decoder_embedding, initial_state=dec1_state_input)
        # layer2
        dec2_input_h = Input(shape=(2*self.latent_dim,))
        dec2_input_c = Input(shape=(2*self.latent_dim,))
        dec2_state_input = [dec2_input_h, dec2_input_c]
        decoder_lstm2 = model.layers[11]
        decoder2_outputs, dec2_state_h, dec2_state_c = decoder_lstm2(decoder1_outputs, initial_state=dec2_state_input)
        # States and Input
        dec_state_input = [dec1_input_h, dec1_input_c, dec2_input_h, dec2_input_c]
        decoder_states = [dec1_state_h, dec1_state_c, dec2_state_h, dec2_state_c]
        encoder2_outputs = Input(shape=(None, 2*self.latent_dim))
        # ATTENTION
        dot_layer1 = model.layers[12]
        attn_inp = dot_layer1([decoder2_outputs, encoder2_outputs])
        act_layer = model.layers[13]
        attn_out = act_layer(attn_inp)
        dot_layer2 = model.layers[14]
        cntxt1 = dot_layer2([attn_out, encoder2_outputs])
        concat_layer = model.layers[15]
        cntxt2 = concat_layer([cntxt1, decoder2_outputs])
        # TIME DISTRIBUTED
        tdt_layer = model.layers[16]
        cntxt3 = tdt_layer(cntxt2)
        tds_layer = model.layers[17]
        output = tds_layer(cntxt3)
        # SET DECODER MODEL
        decoder_model = Model([encoder2_outputs, decoder_inputs] + dec_state_input, [output] + decoder_states)
        model = None
        return encoder_model, decoder_model

class LSTM2_MODEL():
    def __init__(self, data_info, latent_dim=64, embedding_size = 40, dropout=0.1, lr=0.001, attention = False):
        self.description = "LSTM-2-layer-Encoder LSTM-2-layer-Decoder"
        self.attention = attention

        self.max_encoder_seq_length = data_info["max_encoder_seq_length"]
        self.max_decoder_seq_length = data_info["max_decoder_seq_length"]
        self.num_encoder_tokens = data_info["num_encoder_tokens"]
        self.num_decoder_tokens = data_info["num_decoder_tokens"]

        self.latent_dim = latent_dim
        self.dropout = dropout
        self.lr = lr
        self.embedding_size = embedding_size
        self.model = None
        self.name = 'LSTM2'

    def buildModel(self):
        # ENCODER
        encoder_inputs = Input(shape=(None, )) 
        encoder_embedding = Embedding(self.num_encoder_tokens, self.embedding_size, mask_zero = True)(encoder_inputs)
        # layer 1        
        encoder_lstm1 = LSTM(self.latent_dim, return_sequences= True, return_state=True, dropout=self.dropout)    
        encoder1_outputs, enc1_state_h, enc1_state_c = encoder_lstm1(encoder_embedding)
        encoder1_states = [enc1_state_h, enc1_state_c]
        # layer 2
        encoder_lstm2 = LSTM(self.latent_dim, return_sequences=True, return_state=True, dropout=self.dropout)
        encoder2_outputs, enc2_state_h, enc2_state_c = encoder_lstm2(encoder1_outputs)
        encoder2_states = [enc2_state_h, enc2_state_c]
        # encoder states
        encoder_states = [enc1_state_h, enc1_state_c, enc2_state_h, enc2_state_c] 
        # DECODER
        decoder_inputs = Input(shape=(None, ))
        decoder_embedding = Embedding(self.num_decoder_tokens, self.embedding_size,  mask_zero = True)(decoder_inputs)
        # layer 1
        decoder_lstm1 = LSTM(self.latent_dim, return_sequences=True, return_state=True, dropout=self.dropout)
        decoder1_outputs, _, _ = decoder_lstm1(decoder_embedding, initial_state=encoder1_states)
        # layer 2
        decoder_lstm2 = LSTM(self.latent_dim, return_sequences=True, return_state=True, dropout=self.dropout)
        decoder2_outputs, _, _ = decoder_lstm2(decoder1_outputs, initial_state=encoder2_states)
        # Attention
        if self.attention:
            attention = dot([decoder2_outputs, encoder2_outputs], axes=[2, 2])
            attention = Activation('softmax', name='attention')(attention)
            context = dot([attention, encoder2_outputs], axes=[2,1])
            decoder_combined_context = concatenate([context, decoder2_outputs])
            decoder2_outputs = TimeDistributed(Dense(self.embedding_size, activation="tanh"))(decoder_combined_context)
        # dense layer for categorical one-hot output
        outputs = TimeDistributed(Dense(self.num_decoder_tokens, activation="softmax"))(decoder2_outputs)
        # Model Compilation
        adam = optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model = Model([encoder_inputs, decoder_inputs], outputs)
        model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['acc', precision, recall, f1])
        self.model = model

    def loadModel(self, modelPath):
        self.loadConfig(modelPath)
        self.model = load_model(modelPath + '.h5',  custom_objects={'precision': precision, 'recall': recall, 'f1':f1})
        print "Loaded:", modelPath

    def loadConfig(self, filename):
        with open(filename + '_config.json', 'r') as outfile:
           data = json.load(outfile)
        self.latent_dim = data["latent_dim"]
        self.embedding_size = data["embedding_size"]
        self.description = data["description"]
        self.attention = data["attention"]
        self.lr = data["lr"]
        self.dropout = data["dropout"]
        self.max_encoder_seq_length = data["max_encoder_seq_length"]
        self.max_decoder_seq_length = data["max_decoder_seq_length"]
        self.num_encoder_tokens = data["num_encoder_tokens"]
        self.num_decoder_tokens = data["num_decoder_tokens"]
        self.name = data["name"]

    def getDecoder(self, filename):
        model = load_model(filename,  custom_objects={'precision': precision, 'recall': recall, 'f1':f1})
        # ENCODER
        encoder_inputs = model.input[0] #input_1
        _, enc1_state_h, enc1_state_c = model.layers[4].output
        encoder_outputs, enc2_state_h, enc2_state_c = model.layers[6].output
        encoder_states = [enc1_state_h, enc1_state_c, enc2_state_h, enc2_state_c]
        encoder_outputs,_,_ = model.layers[6].output
        # SET ENCODER MODEL
        encoder_model = Model(encoder_inputs,[encoder_outputs] + encoder_states)
        # DECODER
        decoder_inputs = model.input[1] #input_2
        decoder_embedding = model.layers[3](decoder_inputs)
        # layer1
        dec1_input_h = Input(shape=(self.latent_dim,))
        dec1_input_c = Input(shape=(self.latent_dim,))
        dec1_state_input = [dec1_input_h, dec1_input_c]
        decoder_lstm1 = model.layers[5]
        decoder1_outputs, dec1_state_h, dec1_state_c = decoder_lstm1(decoder_embedding, initial_state=dec1_state_input)
        # layer2
        dec2_input_h = Input(shape=(self.latent_dim,))
        dec2_input_c = Input(shape=(self.latent_dim,))
        dec2_state_input = [dec2_input_h, dec2_input_c]
        decoder_lstm2 = model.layers[7]
        decoder2_outputs, dec2_state_h, dec2_state_c = decoder_lstm2(decoder1_outputs, initial_state=dec2_state_input)
        # States and Input
        dec_state_input = [dec1_input_h, dec1_input_c, dec2_input_h, dec2_input_c]
        decoder_states = [dec1_state_h, dec1_state_c, dec2_state_h, dec2_state_c]
        encoder2_outputs = Input(shape=(None, self.latent_dim))
        # ATTENTION
        dot_layer1 = model.layers[8]
        attn_inp = dot_layer1([decoder2_outputs, encoder2_outputs])
        act_layer = model.layers[9]
        attn_out = act_layer(attn_inp)
        dot_layer2 = model.layers[10]
        cntxt1 = dot_layer2([attn_out, encoder2_outputs])
        concat_layer = model.layers[11]
        cntxt2 = concat_layer([cntxt1, decoder2_outputs])
        # TIME DISTRIBUTED
        tdt_layer = model.layers[12]
        cntxt3 = tdt_layer(cntxt2)
        tds_layer = model.layers[13]
        output = tds_layer(cntxt3)
        # SET DECODER MODEL
        decoder_model = Model([encoder2_outputs, decoder_inputs] + dec_state_input, [output] + decoder_states)
        model = None
        return encoder_model, decoder_model
