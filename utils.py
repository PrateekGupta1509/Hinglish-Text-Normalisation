import numpy as np
import pandas as pd
import math

def readData( input_file='data/output.noisy.processed', output_file='data/output.normalized.processed'):
    input_series = pd.read_table(input_file,header=None)
    output_series = pd.read_table(output_file,header=None)

    df = pd.DataFrame()
    df['input']         = pd.Series(input_series[0].str.lower()).astype(str) + '\n'
    df['output']        = '\t' + pd.Series(output_series[0].str.lower()).astype(str) + '\n'
    df['isTest']        = False
    df['bleu_score']    = 0.0
    df['chrf_score']    = 0.0
    df['model_output']  = ""
    df['wer_score']     = 0.0
    # OR USE set(list(string.lowercase)+list(string.digits)+list(string.whitespace)+['\n')
    input_characters        = sorted(set.union(*df.input.map(lambda c: set(list(c)))))
    target_characters       = sorted(set.union(*df.output.map(lambda c: set(list(c)))))

    num_encoder_tokens      = min(len(input_characters),50) + 1
    num_decoder_tokens      = min(len(target_characters),50) + 1

    max_encoder_seq_length  = min(df.input.map(len).max(),100)
    max_decoder_seq_length  = min(df.output.map(len).max(),100)

    input_token_index       = dict([(char, i+1) for i, char in enumerate(input_characters)])
    target_token_index      = dict([(char, i+1) for i, char in enumerate(target_characters)])

    reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

    data_info = {}
    data_info["max_encoder_seq_length"] = max_encoder_seq_length
    data_info["max_decoder_seq_length"] = max_decoder_seq_length
    data_info["input_characters"] = input_characters
    data_info["target_characters"] = target_characters
    data_info["num_encoder_tokens"] = num_encoder_tokens
    data_info["num_decoder_tokens"] = num_decoder_tokens
    data_info["input_token_index"] = input_token_index
    data_info["target_token_index"] = target_token_index
    data_info["reverse_input_char_index"] = reverse_input_char_index
    data_info["reverse_target_char_index"] = reverse_target_char_index

    return df, data_info


def encodeData(df_batch, data_info, encoding='paddNL'):
    num_samples = df_batch.shape[0]
    if encoding == 'padd0':
        encoder_input_data = np.zeros((num_samples, data_info["max_encoder_seq_length"]),dtype='float32')
        decoder_input_data = np.zeros((num_samples, data_info["max_decoder_seq_length"]),dtype='float32')
        decoder_target_data = np.zeros((num_samples, data_info["max_decoder_seq_length"], data_info["num_decoder_tokens"]),dtype='float32')
        count = 0
        for index, row in  df_batch.iterrows():

            for t, char in enumerate(row['input']):
                if t>=data_info["max_encoder_seq_length"]:
                    break
                encoder_input_data[count % num_samples , t] = data_info["input_token_index"][char]

            # decoder_target_data is ahead of decoder_input_data by one timestep
            for t, char in enumerate(row['output']):
                if t >= data_info["max_decoder_seq_length"]:
                    break
                decoder_input_data[count % num_samples, t] = data_info["target_token_index"][char]
                # decoder_target_data will be ahead by one timestep and will not include the start character.
                if t > 0:
                    decoder_target_data[count % num_samples, t - 1, data_info["target_token_index"][char]] = 1.            
            count += 1

    elif encoding == 'paddNL':
        encoder_input_data = np.full((num_samples, data_info["max_encoder_seq_length"]), data_info['input_token_index']['\n'],dtype='float32')
        decoder_input_data = np.full((num_samples, data_info["max_decoder_seq_length"]), data_info['target_token_index']['\n'],dtype='float32')
        decoder_target_data = np.zeros((num_samples, data_info["max_decoder_seq_length"], data_info["num_decoder_tokens"]),dtype='float32')
        count = 0
        for index, row in  df_batch.iterrows():

            for t, char in enumerate(row['input']):
                if t>=data_info["max_encoder_seq_length"]:
                    break
                encoder_input_data[count % num_samples , t] = data_info["input_token_index"][char]

            # decoder_target_data is ahead of decoder_input_data by one timestep
            for t, char in enumerate(row['output']):
                if t >= data_info["max_decoder_seq_length"]:
                    break
                decoder_input_data[count % num_samples, t] = data_info["target_token_index"][char]
                # decoder_target_data will be ahead by one timestep and will not include the start character.
                if t > 0:
                    decoder_target_data[count % num_samples, t - 1, data_info["target_token_index"][char]] = 1.
            # padd with \n character
            index_newline = min(len(row['output']), data_info['max_decoder_seq_length']) - 1
            decoder_target_data[count % num_samples, index_newline: , data_info["target_token_index"]['\n']] = 1.
            count += 1

    return encoder_input_data, decoder_input_data, decoder_target_data


def splitData(df, split_ratio = 0.2, randSeed = 42):
    data_size = len(df.index)
    perm = np.random.RandomState(seed=randSeed).permutation(df.index)

    df.loc[perm[:int((1-split_ratio)*data_size)], ['isTest']] = True

    df_train = df.iloc[perm[:int((1-split_ratio)*data_size)]]
    df_test = df.iloc[perm[int((1-split_ratio)*data_size):]]
    return df_train, df_test


def batchGenerator(df, data_info, batch_size, encoding='paddNL'):
    total_batch = int(math.ceil(float(df.shape[0])/batch_size))
    # print total_batch
    while True:
        for batchNo in range(0,total_batch):
            df_batch = df[ batchNo*batch_size : (batchNo+1)*batch_size]
            # print data_info
            enc_input, dec_input, dec_target = encodeData(df_batch,data_info,encoding)
            # print df_batch.shape, enc_input.shape
            yield ([enc_input, dec_input], dec_target)
