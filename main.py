### Check GPUs
# from tensorflow.python.client import device_lib
# print device_lib.list_local_devices()

from utils import readData, batchGenerator
from model import *
from train import train
from evaluate import decodeSample, evaluate, getVisuals

if __name__ == '__main__':
    # READ DATA
    df, data_info = readData( 'data/output.noisy.processed', 'data/output.normalized.processed')

    filename = 'savedModel/attn_bilstm2'
    modelObj = BiLSTM_MODEL2(data_info,lr=0.005, attention = True)

    # Build Model
    modelObj.buildModel()

    # Load Model
    # modelObj.loadModel(filename)

    # Train Model
    train(filename, modelObj, df, data_info, batchGenerator, 100, 0, 50, encoding="padd0")

    # Decode a Samples
    # decodeSample(filename, modelObj.model, df, data_info, attn = True)

    # Evaluate and Save
    evaluate(modelObj, filename, df[:], data_info, encoding="padd0")

    # Graphs Visual for everything
    # getVisuals(filename)


    # LEARNING CURVE
    # print 1
    # filename = 'savedModel/learncurve/attn_bigru2_10k'
    # modelObj = BiGRU2_MODEL(data_info, attention = True)
    # modelObj.buildModel()
    # train(filename, modelObj, df[:10000], data_info, batchGenerator, 100, 0, 30, encoding="padd0")

    # print 2
    # filename = 'savedModel/learncurve/attn_bigru2_20k'
    # modelObj = BiGRU2_MODEL(data_info, attention = True)
    # modelObj.buildModel()
    # train(filename, modelObj, df[:20000], data_info, batchGenerator, 100, 0, 30, encoding="padd0")

    # print 3
    # filename = 'savedModel/learncurve/attn_bigru2_30k'
    # modelObj = BiGRU2_MODEL(data_info, attention = True)
    # modelObj.buildModel()
    # train(filename, modelObj, df[:30000], data_info, batchGenerator, 100, 0, 30, encoding="padd0")

    # print 4
    # filename = 'savedModel/learncurve/attn_bigru2_40k'
    # modelObj = BiGRU2_MODEL(data_info, attention = True)
    # modelObj.buildModel()
    # train(filename, modelObj, df[:40000], data_info, batchGenerator, 100, 0, 30, encoding="padd0")

    # print 5
    # filename = 'savedModel/learncurve/attn_bigru2_50k'
    # modelObj = BiGRU2_MODEL(data_info, attention = True)
    # modelObj.buildModel()
    # train(filename, modelObj, df[:50000], data_info, batchGenerator, 100, 0, 30, encoding="padd0")

    # print 6
    # filename = 'savedModel/learncurve/attn_bigru2_60k'
    # modelObj = BiGRU2_MODEL(data_info, attention = True)
    # modelObj.buildModel()
    # train(filename, modelObj, df[:60000], data_info, batchGenerator, 100, 0, 30, encoding="padd0")

    # print 7
    # filename = 'savedModel/learncurve/attn_bigru2_70k'
    # modelObj = BiGRU2_MODEL(data_info, attention = True)
    # modelObj.buildModel()
    # train(filename, modelObj, df[:70000], data_info, batchGenerator, 100, 0, 30, encoding="padd0")

    # print 8
    # filename = 'savedModel/learncurve/attn_bigru2_80k'
    # modelObj = BiGRU2_MODEL(data_info, attention = True)
    # modelObj.buildModel()
    # train(filename, modelObj, df[:80000], data_info, batchGenerator, 100, 0, 30, encoding="padd0")

    # print 9
    # filename = 'savedModel/learncurve/attn_bigru2_90k'
    # modelObj = BiGRU2_MODEL(data_info, attention = True)
    # modelObj.buildModel()
    # train(filename, modelObj, df[:90000], data_info, batchGenerator, 100, 0, 30, encoding="padd0")

    # print 10
    # filename = 'savedModel/learncurve/attn_bigru2_100k'
    # modelObj = BiGRU2_MODEL(data_info, attention = True)
    # modelObj.buildModel()
    # train(filename, modelObj, df[:100000], data_info, batchGenerator, 100, 0, 30, encoding="padd0")

    # print 11
    # filename = 'savedModel/learncurve/attn_bigru2_110k'
    # modelObj = BiGRU2_MODEL(data_info, attention = True)
    # modelObj.buildModel()
    # train(filename, modelObj, df[:110000], data_info, batchGenerator, 100, 0, 30, encoding="padd0")

    # print 12
    # filename = 'savedModel/learncurve/attn_bigru2_120k'
    # modelObj = BiGRU2_MODEL(data_info, attention = True)
    # modelObj.buildModel()
    # train(filename, modelObj, df[:120000], data_info, batchGenerator, 100, 0, 30, encoding="padd0")

    # print 13
    # filename = 'savedModel/learncurve/attn_bigru2_130k'
    # modelObj = BiGRU2_MODEL(data_info, attention = True)
    # modelObj.buildModel()
    # train(filename, modelObj, df[:130000], data_info, batchGenerator, 100, 0, 30, encoding="padd0")

    # print 14
    # filename = 'savedModel/learncurve/attn_bigru2_140k'
    # modelObj = BiGRU2_MODEL(data_info, attention = True)
    # modelObj.buildModel()
    # train(filename, modelObj, df[:140000], data_info, batchGenerator, 100, 0, 30, encoding="padd0")

    # print 15
    # filename = 'savedModel/learncurve/attn_bigru2_150k'
    # modelObj = BiGRU2_MODEL(data_info, attention = True)
    # modelObj.buildModel()
    # train(filename, modelObj, df[:150000], data_info, batchGenerator, 100, 0, 30, encoding="padd0")
