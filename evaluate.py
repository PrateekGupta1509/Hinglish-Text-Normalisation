from utils import encodeData
import numpy as np
import pandas as pd
from keras.models import Model
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pandas.plotting import table
import seaborn
from evalMetrics import *

def generateOutput(model, df_batch, data_info, encoding = 'paddNL', attn = False):
    enc_input, dec_input, _ = encodeData(df_batch, data_info, encoding)
    dec_input = np.zeros_like(dec_input)
    dec_input[:,0] = data_info['target_token_index']['\t']

    for i in range(1, dec_input.shape[1]):
        if attn:
            output, attention = model.predict([enc_input, dec_input])
            attn_density = attention
        else:
            output = model.predict([enc_input, dec_input])
        dec_input[:,i] = output.argmax(axis=2)[:,i-1]
        # Break if all row have current character as \n
        if np.all(dec_input[:i] == data_info['target_token_index']['\n']):
            break

    decoded_output = np.vectorize(data_info['reverse_target_char_index'].get)(dec_input[:,1:])
    decoded_output = map(''.join, decoded_output)
    if attn:
        return decoded_output, attn_density
    return decoded_output

def decodeBatch(modelType, enc_model, dec_model, df_batch, data_info, encoding = 'paddNL', attn = False):
    enc_input, _, _ = encodeData(df_batch, data_info, encoding)
    # Run Encoder Model
    if modelType == 'BiLSTM2':
        enc_out, enc1_state_h, enc1_state_c, enc2_state_h, enc2_state_c = enc_model.predict(enc_input)
        states_value = [enc1_state_h, enc1_state_c, enc2_state_h, enc2_state_c]

    elif modelType == 'BiGRU2':
        enc_out, enc1_state_h, enc2_state_h = enc_model.predict(enc_input)
        states_value = [enc1_state_h, enc2_state_h]

    elif modelType == 'LSTM2':
        enc_out, enc1_state_h, enc1_state_c, enc2_state_h, enc2_state_c = enc_model.predict(enc_input)
        states_value = [enc1_state_h, enc1_state_c, enc2_state_h, enc2_state_c]

    elif modelType == 'GRU2':
        enc_out, enc1_state_h, enc2_state_h = enc_model.predict(enc_input)
        states_value = [enc1_state_h, enc2_state_h]

    elif modelType == 'BiLSTM':
        enc_out, enc1_state_h, enc1_state_c = enc_model.predict(enc_input)
        states_value = [enc1_state_h, enc1_state_c]

    elif modelType == 'BiGRU':
        enc_out, enc1_state_h = enc_model.predict(enc_input)
        states_value = [enc1_state_h]

    elif modelType == 'LSTM':
        enc_out, enc1_state_h, enc1_state_c = enc_model.predict(enc_input)
        states_value = [enc1_state_h, enc1_state_c]

    elif modelType == 'GRU':
        enc_out, enc1_state_h = enc_model.predict(enc_input)
        states_value = [enc1_state_h]

    tar_seq = np.zeros((df_batch.shape[0],data_info["max_decoder_seq_length"]+1), dtype='float32')
#     tar_seq = np.zeros((df_batch.shape[0],1))
    tar_seq[:,0] = data_info['target_token_index']['\t']
    counter = 0
    while counter < data_info["max_decoder_seq_length"]: #decoded_sentence[-1:] != '\n' and
        # Run Decoder Model
        if modelType == 'BiLSTM2':
            dec_out, enc1_state_h, enc1_state_c, enc2_state_h, enc2_state_c = dec_model.predict([enc_out, tar_seq[:,counter]]+states_value)
            states_value = [enc1_state_h, enc1_state_c, enc2_state_h, enc2_state_c]

        elif modelType == 'BiGRU2':
            dec_out, enc1_state_h, enc2_state_h = dec_model.predict([enc_out, tar_seq[:,counter]]+states_value)
            states_value = [enc1_state_h, enc2_state_h]

        elif modelType == 'LSTM2':
            dec_out, enc1_state_h, enc1_state_c, enc2_state_h, enc2_state_c = dec_model.predict([enc_out, tar_seq[:,counter]]+states_value)
            states_value = [enc1_state_h, enc1_state_c, enc2_state_h, enc2_state_c]

        elif modelType == 'GRU2':
            dec_out, enc1_state_h, enc2_state_h = dec_model.predict([enc_out, tar_seq[:,counter]]+states_value)
            states_value = [enc1_state_h, enc2_state_h]

        elif modelType == 'BiLSTM':
            dec_out, enc1_state_h, enc1_state_c = dec_model.predict([enc_out, tar_seq[:,counter]]+states_value)
            states_value = [enc1_state_h, enc1_state_c]

        elif modelType == 'BiGRU':
            dec_out, enc1_state_h = dec_model.predict([enc_out, tar_seq[:,counter]]+states_value)
            states_value = [enc1_state_h]

        elif modelType == 'LSTM':
            dec_out, enc1_state_h, enc1_state_c = dec_model.predict([enc_out, tar_seq[:,counter]]+states_value)
            states_value = [enc1_state_h, enc1_state_c]

        elif modelType == 'GRU':
            dec_out, enc1_state_h = dec_model.predict([enc_out, tar_seq[:,counter]]+states_value)
            states_value = [enc1_state_h]

        # Get the current decoded output
        sampled_token_index = dec_out.argmax(axis=2)[:,-1]
        # Update the target sequence (of length 1).
        counter += 1
        tar_seq[:, counter] = sampled_token_index
    # Convert last character to \n for all cases
    tar_seq[:,-1] = data_info['target_token_index']['\n']
    tar_seq = np.vectorize(data_info['reverse_target_char_index'].get)(tar_seq)
    tar_seq = [ ''.join(y[1:np.where(y == '\n')[0][0]]) for y in tar_seq]
    return tar_seq

def visualizeAttn(filename, attention_density, inText, outText, targetText):
    plt.clf()
    fig = plt.figure(figsize=(28,12))
    plt.title("Attention Activation", fontsize = 24)
    ax = seaborn.heatmap(attention_density[:len(outText) + 1, : len(inText) + 1],
        xticklabels=[w for w in inText],
        yticklabels=[w for w in outText])
    ax.invert_yaxis()
    fig.subplots_adjust(bottom=0.15)
    fig.text(0.1, 0.05, 'Input:  '+inText+'\nTarget: '+targetText+'\nOutput: '+outText, fontsize = 18)
    plt.savefig(filename, dpi = 150)
    # plt.show()

def decodeSample(filename, model, df, data_info, encoding = 'paddNL', attn = False, n = 5):
    if isinstance(df, list):
        dfs = pd.DataFrame(df, columns = ['input'])
        dfs['output'] = ''
    elif not isinstance(n, int):
        dfs = df.loc[n]
    else:
        dfs = df.sample(n)
    print dfs
    if attn:
        attn_layer = model.get_layer('attention')
        attn_model = Model(inputs=model.inputs, outputs=model.outputs + [attn_layer.output])
        outputs, attn_density = generateOutput(attn_model, dfs, data_info, encoding = 'paddNL', attn = True)
    else:
        outputs = generateOutput(model, dfs, data_info, encoding = 'paddNL')

    for i in range(dfs.shape[0]):
        inText = dfs.iloc[i,0].strip('\t,\n')
        targetText = dfs.iloc[i,1].strip('\t,\n')
        outText = ''.join(outputs[i])
        try:
            outText = outText[ :outText.index('\n')]
        except:
            pass
        print "\nINPUT:", inText
        print "TARGET:", targetText
        print "MODEL:", outText
        if attn:
            visualizeAttn(filename+str(dfs.index[i]), attn_density[i], inText, outText, targetText)

def evaluate(modelObj, filename, df, data_info, encoding = 'paddNL', batch_size = 10000):
    a,b = modelObj.getDecoder(filename+'.h5')
    # smooth_func = bleu_score.SmoothingFunction()
    for batch_no in range(0,df.shape[0],batch_size):
        print "Evaluating ", batch_no,"/", df.shape[0]
        df_batch = df[batch_no: (batch_no+batch_size)]
        df_batch['model_output'] = decodeBatch(modelObj.name, a, b, df_batch, data_info, encoding)
    # strip all \t and \n
    df['input'] = df['input'].str.strip('\t,\n')
    df['output'] = df['output'].str.strip('\t,\n')
    df['model_output'] = df['model_output'].str.strip('\t,\n')
    # get all scores
    df['chrf_score'] = df[["output","model_output"]].apply(lambda x: chrf(x[0].split(), x[1].split()), axis=1)
    df['bleu_score'] = df[["output","model_output"]].apply(lambda x: bleu([x[0].split()], x[1].split()), axis=1)
    df['wer_score'] = df[["output","model_output"]].apply(lambda x: wer(x[0].split(), x[1].split()), axis=1)
    # print df['model_output']
    df.to_csv(filename+"_output.csv")
    print "Saved evaluation output csv"


def generateGraph( X, Y1, Y2 , title, xLabel, yLabel):
    plt.clf()
    fig = plt.figure()
    plt.plot(X, Y1)
    if len(Y2) != 0:
        plt.plot(X, Y2)
    plt.title(title)
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.legend(['train', 'val'], loc='upper left')
    fileName = title+'.png'
    plt.savefig(fileName,dpi = 150)
    # plt.show()

def getVisuals(filename):
    logs = pd.read_csv(filename + '_logs.csv', index_col = [0])
    for i in ['loss','acc','precision','recall','f1']:
        generateGraph( logs['eps'], logs[i], logs['val_'+i], filename+'_'+i , 'Epochs', i)
    generateGraph(logs['eps'], logs['ts'], [],'Model-Epoch-TimeStamp', 'Epochs', 'Time-Taken')

    dfl = pd.read_csv(filename + '_output.csv', index_col = [0])
    ax = plt.subplot(111, frame_on=False) # no visible frame
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis
    table(ax, dfl)  # where df is your data frame
    plt.savefig(filename+'table.png')
