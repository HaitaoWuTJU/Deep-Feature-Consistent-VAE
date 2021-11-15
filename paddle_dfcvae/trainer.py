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
import numpy as np
from PIL import Image
# def setdir(path):
#     import shutil
#     if os.path.exists(path):
#         shutil.rmtree(path)
#     os.makedirs(path)


#
# warnings.filterwarnings('ignore')


class Trainer():
    def __init__(self, expName, resume=True, resume_inter='latest', CONFIG=None):
        super(Trainer, self).__init__()

        self.batch_size = CONFIG['exp_params']['batch_size']


        self.config = CONFIG
        self.root = os.getcwd()
        self.resume = resume
        self.resume_inter = resume_inter
        self.best_avgAcc = -1.0
        self.best_avgAcc_inter = -100
        self.exp_dir = self.init_exp_dir(expName=expName)
        self.logger = self.init_logger()
        self.models = self.init_models()
        self.dataloaders = self.init_dataloaders()
        self.optimizers = self.init_optimizers()
        self.criterions = self.init_criterions()
        self.init_check()
        self.stepEachEpoch= len(self.dataloaders['train']())
        self.max_inter = 50 * self.stepEachEpoch
        self.inter = 1
        self.latent_dim=128
        self.train_display_step = 100
        self.train_save_step = 300
        self.val_step = 500
        self.imgNum = self.img_train_cnt
        self.logger.info('----init complete----')
        self.logger.info('max_inter:{},img_num:{}'.format(self.max_inter, self.imgNum))
    def run(self):
        ##resume
        if self.resume:
            self.resume_experiment()
        self.Loss = {}
        self.Loss['step_cnt'] = 0
        self.Loss['loss'] = 0.0
        self.Loss['Reconstruction_Loss'] = 0.0
        self.Loss['KLD'] = 0.0
        self.Loss['weight_KLD']=0.0
        self.Loss['feature_loss'] =0.0
        while (self.inter < self.max_inter):
            for batch_id, batch in enumerate(tqdm(self.dataloaders['train']())):
                self.change_learing_rate()
                lossAll = self.train_step(batch_id, batch)
                self.Loss['step_cnt'] += 1
                self.Loss['loss'] += lossAll['loss']
                # self.Loss['Reconstruction_Loss'] += lossAll['Reconstruction_Loss'].numpy()
                self.Loss['KLD'] += lossAll['KLD']
                # self.Loss['weight_KLD'] += lossAll['weight_KLD'].numpy()
                self.Loss['feature_loss'] += lossAll['feature_loss']
                if self.inter % self.train_display_step == 0:
                    infoStrEasy='inter:{} , loss:{},feature_loss:{}, KLD_loss:{},lr:{}' \
                                     .format(self.inter,
                                             self.Loss['loss'] / self.Loss['step_cnt'],
                                             self.Loss['feature_loss'] / self.Loss['step_cnt'],
                                             self.Loss['KLD'] / self.Loss['step_cnt'],
                                             self.optimizers['dfcvae'].get_lr(),
                                             )
                    self.logger.info(infoStrEasy)
                    self.Loss['step_cnt'] = 0
                    self.Loss['loss'] = 0.0
                    # self.Loss['Reconstruction_Loss'] = 0.0
                    self.Loss['KLD'] = 0.0
                    # self.Loss['weight_KLD'] = 0.0
                    self.Loss['feature_loss'] = 0.0
                if self.inter % self.train_save_step == 0:
                    self.save_checkpoint()
                    self.delete_checkpoint()

                if self.inter % self.val_step == 0:
                    self.valuate()
                self.inter += 1

    def resume_experiment(self):
        self.load_checkpoint()

    def change_learing_rate(self):
        init_lr=self.config['exp_params']['LR']
        decay = 99.0 / self.max_inter

        # lr_dfcvae = init_lr*math.pow(0.5,current_epoch)
        lr_dfcvae=init_lr/(1 + decay * self.inter) #start 0
        self.optimizers['dfcvae'].set_lr(lr_dfcvae)

    def init_exp_dir(self, expName):
        expdir = os.path.join(self.root, 'experiments', expName)
        if os.path.exists(expdir) and not self.resume:
            # now_str = datetime.datetime.now().__str__().replace(' ','_')
            # expdir=os.path.join(self.root,'experiments',expName+'_'+now_str)
            # if os.path.exists(expdir):
            print('exist exp dir {}'.format(expdir))
            exit(1)
            # else:
            #     os.makedirs(expdir)
        else:

            os.makedirs(expdir, exist_ok=True)
            models_dir = os.path.join(expdir, 'ckpt')
            os.makedirs(models_dir, exist_ok=True)
            imgRec_dir=os.path.join(expdir, 'reconstruction')
            os.makedirs(imgRec_dir, exist_ok=1)
            img_dir = os.path.join(expdir, 'generated')
            os.makedirs(img_dir, exist_ok=1)
        return expdir

    def init_logger(self):
        logger = logging.getLogger(__name__)
        strHandler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)-8s - %(levelname)-6s - %(message)s')
        strHandler.setFormatter(formatter)
        logger.addHandler(strHandler)
        logger.setLevel(logging.INFO)
        log_dir = os.path.join(self.exp_dir, 'logs')
        if (not os.path.isdir(log_dir)):
            os.makedirs(log_dir)
        now_str = datetime.datetime.now().__str__().replace(' ', '_')
        self.log_file = os.path.join(log_dir, 'LOG_INFO_' + now_str + '.txt')
        self.log_fileHandler = logging.FileHandler(self.log_file)
        self.log_fileHandler.setFormatter(formatter)
        logger.addHandler(self.log_fileHandler)
        return logger

    def init_models(self):
        dfcvae, msg_resnet = get_model()
        networks = {}
        networks['dfcvae'] = dfcvae
        self.logger.info(msg_resnet)
        return networks  ##['backbone','APCHead','FCNHead']

    def init_dataloaders(self):
        # trainLoader,valLoader,testLoader
        trainLoader, testLoader,img_train_cnt ,msg= get_dataloader()
        self.img_train_cnt=img_train_cnt
        dataloaders = {}
        dataloaders['train'] = trainLoader
        dataloaders['test'] = testLoader
        return dataloaders

    def init_optimizers(self):
        optimizers = {}
        optimizers['dfcvae'] = get_optimizers(self.models['dfcvae'],config=self.config['exp_params'])

        return optimizers

    def init_criterions(self):
        criterions = {}
        criterions['MSELoss'] = paddle.nn.MSELoss()
        return criterions

    def init_check(self):
        assert len(self.models) == 1
        assert len(self.optimizers) == 1
        assert len(self.criterions) == 1

    def train_step(self, batch_id, batch):
        self.models['dfcvae'].train()
        self.optimizers['dfcvae'].clear_grad()

        x, _ = batch
        outs = self.models['dfcvae'](x)
        # lossAll=self.models['dfcvae'].loss_function(outs)
        lossAll = self.compute_loss(outs)

        lossAll['loss'].backward()


        self.optimizers['dfcvae'].step()
        lossAll['loss']=lossAll['loss'].numpy()
        return lossAll  # (apc,fcn)

    def compute_loss(self, outputs):
        recons = outputs[0]
        input = outputs[1]
        recons_features = outputs[2]
        input_features = outputs[3]
        mu = outputs[4]
        log_var = outputs[5]
        # self.params['batch_size'] / self.num_train_imgs
        kld_weight = self.batch_size / self.imgNum  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        feature_loss = 0.0
        for (r, i) in zip(recons_features, input_features):
            # feature_loss += self.criterions['MSELoss'](r, i)
            feature_loss += F.mse_loss(r, i)

        ##paddle klloss
        # kld_loss=F.kl_div(mu,log_var, reduction='mean')

        ##pyorch-vae
        kld_loss =kld_weight*paddle.mean(-0.5 * paddle.sum(1 + log_var - mu ** 2 - log_var.exp(), axis=1), axis=0)

        ##anonter
        # kld_loss = -0.5*paddle.mean(1 + log_var - mu ** 2 - log_var.exp())

        # beta = 0.5
        # alpha=1.0
        # loss = beta * (recons_loss + feature_loss) + alpha * kld_loss
        # loss = 0.5 * (recons_loss + feature_loss) + 1.0 * kld_weight * kld_loss

        # loss = 0.5 * feature_loss + 1.0 *kld_loss
        loss = 0.5 * (recons_loss+ feature_loss) + 1.0 * kld_loss

        return {'loss': loss,'feature_loss':feature_loss.numpy() ,'KLD': kld_loss.numpy()}

    def valuate(self):
        self.models['dfcvae'].eval()
        # img_dir = os.path.join(self.exp_dir, 'generated')
        img_dir = os.path.join(self.exp_dir, 'reconstruction')
        save_path=os.path.join(img_dir,f'{self.inter}.png')
        row1Img = []
        row2Img = []
        for batch_id, batch in enumerate(tqdm(self.dataloaders['test'])):
            x, _ = batch
            outputs = self.models['dfcvae'](x)
            sourceImg = np.array((x[0] + 1) / 2 * 255, dtype='uint8').transpose([1, 2, 0])

            reconImg = np.array((outputs[0][0] + 1) / 2 * 255, dtype='uint8').transpose([1, 2, 0])

            row1Img.append(sourceImg)
            row2Img.append(reconImg)

            if batch_id > 10:
                break

        row1Img = np.concatenate(row1Img, axis=1)
        row2Img = np.concatenate(row2Img, axis=1)
        img = np.concatenate([row1Img, row2Img], axis=0)
        img = Image.fromarray(img)
        img.save(save_path)


        z = np.random.rand(10, self.latent_dim)
        z = paddle.to_tensor(z, dtype='float32')
        res = self.models['dfcvae'].decode(z)
        saveDir =os.path.join(self.exp_dir,'generated') #f'./experiments/{expname}/generated'

        imgs = []
        for i in range(res.shape[0]):
            genImg = np.array((res[i] + 1) / 2 * 255, dtype='uint8').transpose([1, 2, 0])
            imgs.append(genImg)
        imgs = np.concatenate(imgs, axis=1)
        imgs = Image.fromarray(imgs)
        imgs.save(os.path.join(saveDir, f'generated_{self.inter}.png'))



    def save_checkpoint(self, postfix='normal'):
        state = {}
        state['inter'] = self.inter
        state['best'] = {}
        state['best']['best_avgAcc'] = self.best_avgAcc
        state['best']['best_avgAcc_inter'] = self.best_avgAcc_inter
        state['models'] = {}
        state['models']['dfcvae'] = self.models['dfcvae'].state_dict()

        state['optimizers'] = {}
        state['optimizers']['dfcvae'] = self.optimizers['dfcvae'].state_dict()

        if postfix == 'best':
            save_path = os.path.join(self.exp_dir, 'ckpt',
                                     str(self.inter) + '.pdparams.bestAvgAcc_{}'.format(self.best_miou))
            paddle.save(state, save_path)
            self.logger.info('save best ckpt inter:{}, best avgAcc : {}'.format(self.inter, self.best_miou))
        else:
            save_path = os.path.join(self.exp_dir, 'ckpt', str(self.inter) + '.pdparams')
            paddle.save(state, save_path)
            self.logger.info('save ckpt inter:{}'.format(self.inter))

    def delete_checkpoint(self):
        ckpts = list(glob.glob(self.exp_dir + '/ckpt/*.pdparams'))
        for ckpt in ckpts:
            inter = int(ckpt.split('/')[-1].split('.')[0])
            if inter != self.inter:
                delete_path = os.path.join(self.exp_dir, 'ckpt', str(inter) + '.pdparams')
                os.remove(delete_path)

    def load_checkpoint(self):
        if self.resume_inter == 'latest':

            ckpts = list(glob.glob(self.exp_dir + '/ckpt/*.pdparams'))
            latest = -1
            for ckpt in ckpts:
                inter = int(ckpt.split('/')[-1].split('.')[0])
                if inter > latest:
                    latest = inter
            load_path = os.path.join(self.exp_dir, 'ckpt', str(latest) + '.pdparams')
            pass
        else:
            load_path = os.path.join(self.exp_dir, 'ckpt', str(self.resume_inter) + '.pdparams')
        # print(load_path)
        try:
            state = paddle.load(load_path)
        except:
            if latest == -1:
                self.logger.info("no ckpt, no load ckpt")
                return
            os.remove(load_path)
            ckpts = list(glob.glob(self.exp_dir + '/ckpt/*.pdparams'))
            latest = -1
            for ckpt in ckpts:
                inter = int(ckpt.split('/')[-1].split('.')[0])
                if inter > latest:
                    latest = inter
            load_path = os.path.join(self.exp_dir, 'ckpt', str(latest) + '.pdparams')
            state = paddle.load(load_path)
        self.inter = state['inter'] + 1
        self.best_avgAcc = state['best']['best_avgAcc']
        self.best_avgAcc_inter = state['best']['best_avgAcc_inter']
        self.models['dfcvae'].set_state_dict(state['models']['dfcvae'])

        self.optimizers['dfcvae'].set_state_dict(state['optimizers']['dfcvae'])

        self.logger.info('resume ckpt from {}'.format(load_path))


if __name__ == '__main__':
    with open('./dfc_vae.yaml', 'r') as file:
        CONFIG = yaml.safe_load(file)
    # expname='dfcvae-latent-100-conv-featureLoss_KLdLoss'
    expname='vae123_conv_his_loss'
    expname = 'vae345_conv_his_loss'
    trainer = Trainer(expName=expname, resume =True, resume_inter='latest', CONFIG=CONFIG)
    trainer.run()