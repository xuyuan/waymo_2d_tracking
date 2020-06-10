# launch with `streamlit run demo.py`
import streamlit as st


"""
# Cervical Cancer Detection
"""

import torch
import numpy as np
from data import create_dataset
from inference import load_model
from utils.visualization import draw_train_sample, draw_detection

from types import SimpleNamespace
args = SimpleNamespace(models=['best_metric.model.pth'],
                       data_root='/media/data/CervicalCancer')

gpu_or_cup = st.sidebar.selectbox('device', ('CPU', 'GPU'), index=int(torch.cuda.is_available()))


@st.cache(allow_output_mutation=True)
def create_dataset_cached(*args, **kwargs):
    return create_dataset(*args, **kwargs)


@st.cache(allow_output_mutation=True)
def load_model_cached(model_file):
    model = load_model(model_file)
    model.eval()
    return model

model = load_model_cached(args.models[0])

if gpu_or_cup == 'GPU':
    model = model.cuda()
else:
    model = model.cpu()

image_size = st.sidebar.selectbox('image size', (800, 1024))
data_fold = st.sidebar.selectbox('data fold', range(5))
dataset = create_dataset_cached(args.data_root, mode='test', data_fold=data_fold, image_size=(image_size, image_size))
print(dataset)

sample_id = st.sidebar.selectbox('Please select one sample', range(len(dataset)))
if st.sidebar.button('random'):
    sample_id = np.random.choice(len(dataset))


sample = dataset[sample_id]
image = sample['input']
bbox = sample['bbox']


image_gt = draw_train_sample(sample)

conf_thresh = st.sidebar.slider('confidence', 0., 1., .05)
nms_thresh = st.sidebar.slider('NMS', 0., 1., .15)
d = model.predict(image, conf_thresh=conf_thresh, nms_thresh=nms_thresh)
image_gt = draw_detection(image_gt, d, dataset.classnames)
st.image(image_gt, caption=f'Prediction #{sample_id}: {sample["image_id"]} size={image.size} bbox={bbox}', use_column_width=True)
