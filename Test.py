import sys
import numpy as np
import librosa
from keras.models import load_model

model=load_model('HVDv1.h5')
audio,sr=librosa.load(sys.argv[1])
features=librosa.core.stft(audio,n_fft=1024,hop_length=256,win_length=1024)
features=np.expand_dims(features,axis=-1)
features=np.expand_dims(features,axis=0)
features=np.expand_dims(features,axis=0)
pred=model.predict_classes(features)[0][0]

if pred==1:
 print("Voice Activity Exist.")
else:
 print("No vocal activity exist.")
