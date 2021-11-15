import math

from tqdm import tqdm
import paddle
import datetime
import logging
import os
from dataloader import get_dataloader
from architectures import get_model
from optimizer import get_optimizers
import paddle.nn.functional as F
import warnings
import glob
import yaml
from PIL import Image
from dataloader import get_dataloader
import numpy as np


def setdir(path):
    import shutil
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def generated(expname,inter):
    path = f'./experiments/{expname}/ckpt/{inter}.pdparams'
    state = paddle.load(path)['models']['dfcvae']
    model, msg = get_model()
    model.set_state_dict(state)
    model.eval()
    z=np.random.rand(100,128)
    z=paddle.to_tensor(z,dtype='float32')
    res=model.decode(z)
    saveDir = f'./imgGenerateResult/{expname}_{inter}'
    os.makedirs(saveDir)
    for i in range(res.shape[0]):
        genImg = np.array((res[i] + 1) / 2 * 255, dtype='uint8').transpose([1, 2, 0])
        print(genImg.shape)
        img=Image.fromarray(genImg)
        img.save(os.path.join(saveDir, f'{i}.png'))

def recons(expname, inter=1800):
    path = f'./experiments/{expname}/ckpt/{inter}.pdparams'
    state = paddle.load(path)['models']['dfcvae']
    model, msg = get_model()
    model.set_state_dict(state)
    model.eval()
    trainDataloader, valDataloader, img_cnt, msg = get_dataloader()

    saveDir = f'./tmp/reconstruct_{expname}/'
    # os.makedirs(saveDir,exist_ok=1)
    setdir(saveDir)

    row1Img = []
    row2Img = []

    for batch_id, batch in enumerate(tqdm(valDataloader())):

        x, _ = batch
        output = model(x)
        sourceImg = np.array((x[0] + 1) / 2 * 255, dtype='uint8').transpose([1, 2, 0])

        reconImg = np.array((output[0][0] + 1) / 2 * 255, dtype='uint8').transpose([1, 2, 0])

        row1Img.append(sourceImg)
        row2Img.append(reconImg)

        if batch_id % 10 == 0:
            row1Img = np.concatenate(row1Img, axis=1)
            row2Img = np.concatenate(row2Img, axis=1)
            img = np.concatenate([row1Img, row2Img], axis=0)
            img = Image.fromarray(img)
            img.save(os.path.join(saveDir, f'reconstruct_{batch_id}.png'))

            row1Img = []
            row2Img = []




def generated_a_Pic(expname,inter):
    path = f'./experiments/{expname}/ckpt/{inter}.pdparams'
    state = paddle.load(path)['models']['dfcvae']
    model, msg = get_model()
    model.set_state_dict(state)
    model.eval()

    saveDir = f'./tmp/generated'
    setdir(saveDir)
    # os.makedirs(saveDir, exist_ok=1)
    for index in range(100):
        z = np.random.rand(10, 128)
        z = paddle.to_tensor(z, dtype='float32')
        res = model.decode(z)

        imgs=[]
        for i in range(res.shape[0]):
            genImg = np.array((res[i] + 1) / 2 * 255, dtype='uint8').transpose([1, 2, 0])
            imgs.append(genImg)
        imgs=np.concatenate(imgs,axis=1)
        imgs = Image.fromarray(imgs)
        imgs.save(os.path.join(saveDir,f'generated_{index}.png'))
if __name__ == '__main__':


    expname = 'vae123_conv_his_loss'
    recons(expname, inter=129600)


    # expname = 'vae345_conv_his_loss'
    # recons(expname, inter=127800)
    # generated(expname,129600)
    # generated_a_Pic(expname,129600)
    # generated(expname=expname, inter=27000)
