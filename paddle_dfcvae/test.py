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
def test(expname,inter):
    saveDir=f'./imgGenerateResult/{expname}_{inter}'
    setdir(saveDir)
    path=f'./experiments/{expname}/ckpt/{inter}.pdparams'
    state=paddle.load(path)['models']['dfcvae']
    model,msg=get_model()
    model.set_state_dict(state)
    trainDataloader,valDataloader,img_cnt,msg=get_dataloader()

    for batch_id, batch in enumerate(tqdm(valDataloader())):
        x,_=batch
        output=model(x)
        sourceImg=np.array((x[0]+1)/2*255,dtype='uint8')
        image=Image.fromarray(sourceImg.transpose([1,2,0]))
        image.save(os.path.join(saveDir,f'{batch_id}.png'))
        # reconImg=np.array(output+1*255,dtype='uint8') #
        reconImg = np.array((output[0][0]+1)/2*255,dtype='uint8').transpose([1,2,0])
        print(reconImg[:, :, 2])
        image = Image.fromarray(reconImg)
        image.save(os.path.join(saveDir,f'{batch_id}-rec.png'))
        if batch_id>100:
            break


if __name__=='__main__':
    expname='vae345_his_loss'
    test(expname=expname,inter=28500)