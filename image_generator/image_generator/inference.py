from __future__ import print_function

from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from image_generator.DAMSM import RNN_ENCODER,CustomLSTM
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.autograd import Variable
from image_generator.datasets import TextDataset
from image_generator.miscc.config import cfg

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from image_generator.model import NetG,NetD
import torchvision.utils as vutils

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

def run_inference(caption):
    # Load generator
    lstm = CustomLSTM(256, 256)
    netG = NetG(64, 100, lstm).to('cpu')
    netG.load_state_dict(torch.load('../models/NETG_1.pth', map_location=torch.device('cpu')))

    # Load text encoder
    text_encoder = RNN_ENCODER(5450, nhidden=256)
    state_dict = torch.load('../bird/text_encoder200.pth', map_location=torch.device('cpu'))
    text_encoder.load_state_dict(state_dict)

    dataset = TextDataset('../data/birds', 'test',
                                base_size=cfg.TREE.BASE_SIZE)
    wordtoix = dataset.wordtoix
    sentences = caption.split(',')[:1] # one caption per image

    # a list of indices for a sentence
    captions = []
    cap_lens = []
    for sent in sentences:
        if len(sent) == 0:
            continue
        sent = sent.replace("\ufffd\ufffd", " ")
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(sent.lower())
        if len(tokens) == 0:
            print('sent', sent)
            continue

        rev = []
        for t in tokens:
            t = t.encode('ascii', 'ignore').decode('ascii')
            if len(t) > 0 and t in wordtoix:
                rev.append(wordtoix[t])
        captions.append(rev)
        cap_lens.append(len(rev))

    max_len = np.max(cap_lens)

    sorted_indices = np.argsort(cap_lens)[::-1]
    cap_lens = np.asarray(cap_lens)
    cap_lens = cap_lens[sorted_indices]
    cap_array = np.zeros((len(captions), max_len), dtype='int64')
    for i in range(len(captions)):
        idx = sorted_indices[i]
        cap = captions[idx]
        c_len = len(cap)
        cap_array[i, :c_len] = cap

    # Get number of hidden layers text encoder
    hidden = text_encoder.init_hidden(len(cap_array))

    # Embed text
    words_embs, sent_emb = text_encoder(Variable(torch.from_numpy(np.array(cap_array))), Variable(torch.from_numpy(np.array(cap_lens))), hidden)

    # Generate images
    with torch.no_grad():
        noise = torch.randn(len(cap_array), 100)
        noise = noise.to('cpu')
        netG.lstm.init_hidden(noise)
        fake_imgs = netG(noise, sent_emb)

    # Save generated images
    for j in range(len(fake_imgs)):
        im = fake_imgs[j].data.cpu().numpy()
        # [-1, 1] --> [0, 255]
        im = (im + 1.0) * 127.5
        im = im.astype(np.uint8)
        im = np.transpose(im, (1, 2, 0))
        im = Image.fromarray(im)
        fullpath = f"static/{j}.png"
        im.save(fullpath)