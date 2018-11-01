# Hinglish-Text-Normalisation

The respository contains serveral encoder decoder model with Luong's Attention for Hinglish text normalization.
Since the data was propriety, this repository includes just the models as samples.

### Setting up environment
```
virtualenv venv
source <path-to-venv>/bin/activate
pip install requirements.txt
```
### Launching Flask API
```
python app.py
```
### Training
```
df, data_info = readData( 'data/output.noisy.processed', 'data/output.normalized.processed')
filename = <prefix-path-to-model-storage>
modelObj = <MODEL_CLASS>(data_info, attention = True)
modelObj.buildModel()
train(filename, modelObj, df[0:10], data_info, batchGenerator, 100, 0, 5, encoding="padd0")
```
Encoding can be "padd0" or "paddNL"

### Testing
```
df, data_info = readData( 'data/output.noisy.processed', 'data/output.normalized.processed')
filename = <prefix-path-to-model-storage>
modelObj = <MODEL_CLASS>(data_info, attention = True)
modelObj.loadModel(filename)
evaluate(modelObj, filename, df, data_info, encoding="padd0", batch_size=5000)
```

### Sampling
```
df, data_info = readData( 'data/output.noisy.processed', 'data/output.normalized.processed')
filename = <prefix-path-to-model-storage>
modelObj = <MODEL_CLASS>(data_info, attention = True)
modelObj.loadModel(filename)
decodeSample(filename, modelObj, <DATAFRAME|LIST-OF-STRINGs|LIST-OF-DF-INDICES>, data_info, encoding="padd0", batch_size=5000)
```