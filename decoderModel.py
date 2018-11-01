import numpy as np
from keras.models import Model, load_model
from keras.layers import Input
from metrics import *
from utils import encodeData

def getDecoder(filename, latent_dim = 64):        
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
    dec1_input_h = Input(shape=(2*latent_dim,))
    dec1_state_input = [dec1_input_h]
    decoder_gru1 = model.layers[7]
    decoder1_outputs, dec1_state_h = decoder_gru1(decoder_embedding, initial_state=dec1_state_input)
    # layer2
    dec2_input_h = Input(shape=(2*latent_dim,))
    dec2_state_input = [dec2_input_h]
    decoder_gru2 = model.layers[9]
    decoder2_outputs, dec2_state_h = decoder_gru2(decoder1_outputs, initial_state=dec2_state_input)
    # States and Input
    dec_state_input = [dec1_input_h, dec2_input_h]
    decoder_states = [dec1_state_h, dec2_state_h]
    encoder2_outputs = Input(shape=(None, 2*latent_dim))
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

def decodeSentence(enc_model, dec_model, df_batch, data_info, attn = False):
    enc_input, _, _ = encodeData(df_batch, data_info)
    # Run Encoder Model
    enc_out, enc_state1, enc_state2 = enc_model.predict(enc_input)
    states_value = [enc_state1, enc_state2]
    # Initialize Decoder Input
    decoded_sentence = ''
    tar_seq = np.zeros((1,1))
    tar_seq[0,0] = data_info['target_token_index']['\t']
    
    while decoded_sentence[-1:] != '\n' and len(decoded_sentence) <= data_info["max_decoder_seq_length"]:
        # Run Decoder Model
        dec_out, dec_state1, dec_state2 = dec_model.predict([enc_out, tar_seq]+states_value)
        states_value = [dec_state1, dec_state2]
        # Get the current decoded output
        sampled_token_index = dec_out.argmax(axis=2)[-1,-1]
        decoded_sentence += data_info["reverse_target_char_index"][sampled_token_index]
        # Update the target sequence (of length 1).
        tar_seq[0, 0] = data_info["target_token_index"][decoded_sentence[-1:]]
    return decoded_sentence

