from __future__ import print_function

from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from image_generator.DAMSM import RNN_ENCODER,CustomLSTM
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.autograd import Variable

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

    # Prepare caption
    # caption = 'a red warbler'
    cap = caption.replace("\ufffd\ufffd", " ")
    # picks out sequences of alphanumeric characters as tokens
    # and drops everything else
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(cap.lower())
    tokens_new = []
    for t in tokens:
        t = t.encode('ascii', 'ignore').decode('ascii')
        if len(t) > 0:
            tokens_new.append(t)   
    all_captions = [tokens_new]

    # Build dictionary
    word_counts = defaultdict(float)
    for sent in all_captions:
        for word in sent:
            word_counts[word] += 1

    vocab = [w for w in word_counts if word_counts[w] >= 0]

    ixtoword = {}
    ixtoword[0] = '<end>'
    wordtoix = {}
    wordtoix['<end>'] = 0
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    captions_new = []
    for t in all_captions:
        rev = []
        for w in t:
            if w in wordtoix:
                rev.append(wordtoix[w])
        captions_new.append(rev)

    # Get caption lengths
    captions_lens = []
    for i in captions_new:
        sent_caption = np.asarray(i).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        captions_lens.append(len(sent_caption))
    captions_lens = torch.from_numpy(np.array(captions_lens))
    # sorted_cap_lens, sorted_cap_indices = torch.sort(captions_lens, 0, True)

    # Get number of hidden layers text encoder
    hidden = text_encoder.init_hidden(len(captions_new))

    # Embed text
    words_embs, sent_emb = text_encoder(Variable(torch.from_numpy(np.array(captions_new))), Variable(captions_lens), hidden)

    # Generate images
    with torch.no_grad():
        noise = torch.randn(len(captions_new), 100)
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