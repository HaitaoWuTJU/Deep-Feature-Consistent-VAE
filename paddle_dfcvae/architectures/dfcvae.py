# import torch
from .base import BaseVAE
from .vgg import vgg19_bn
from .types_ import *
# from base import BaseVAE
# from vgg import vgg19_bn
# from types_ import *
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class DFCVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 alpha: float = 1,
                 beta: float = 0.5):
        super(DFCVAE, self).__init__()

        self.latent_dim = latent_dim
        self.alpha = alpha
        self.beta = beta

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2D(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2D(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim, bias_attr=True)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim, bias_attr=True)

        # Build Decoder
        modules = []

        # self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4, bias_attr=True)

        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Conv2DTranspose(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1,
                                       bias_attr=True),
                    nn.BatchNorm2D(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.Conv2DTranspose(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1,
                               bias_attr=True),
            nn.BatchNorm2D(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2D(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh())

        self.feature_network = vgg19_bn()

        # Freeze the pretrained feature network
        for param in self.feature_network.parameters():
            # print('----------------')
            # param.requires_grad = False 不报错
            param.trainable = False
            # print('----------------')

        self.feature_network.eval()

    def encode(self, input):
        result = self.encoder(input)  # 144 512，2，2
        result = paddle.flatten(result, start_axis=1)  # 144 2048
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z):
        result = self.decoder_input(z)
        # result = result.view(-1, 512, 2, 2)
        result = paddle.reshape(result, [-1, 512, 2, 2])  # todo   view>reshape
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar, ):
        std = paddle.exp(0.5 * logvar)
        # eps = torch.randn_like(std)
        eps = paddle.randn(std.shape)
        return eps * std + mu

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)

        # recons_features = self.extract_features(recons,['16','29','42'])
        # input_features = self.extract_features(input,['16','29','42'])
        recons_features = self.extract_features(recons)
        input_features = self.extract_features(input)
        return [recons, input, recons_features, input_features, mu, log_var]

    def extract_features(self,
                         input,
                         feature_layers=None):

        if feature_layers is None:
            # feature_layers = ['0', '7', '14'] ##123 conv
            # feature_layers = ['2', '9', '16']
            #feature_layers = ['16', '29', '42']
            feature_layers =['14', '27', '40'] ##345 conv
            # feature_layers = ['14', '24', '34', '43']
        features = []
        result = input
        # for (key, module) in self.feature_network.features._sub_layers.items():
        #     print(key,module)
        for (key, module) in self.feature_network.features._sub_layers.items():  # todo  _modules>_sub_layers
            result = module(result)
            if key in feature_layers:
                features.append(result)
        return features

    def loss_function(self, outputs):

        recons = outputs[0]
        input = outputs[1]
        recons_features = outputs[2]
        input_features = outputs[3]
        mu = outputs[4]
        log_var = outputs[5]

        kld_weight = 144 / (1131 * 144)  # kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        feature_loss = 0.0
        for (r, i) in zip(recons_features, input_features):
            feature_loss += F.mse_loss(r, i)

        kld_loss = paddle.mean(-0.5 * paddle.sum(1 + log_var - mu ** 2 - log_var.exp(), axis=1), axis=0)

        loss = self.beta * (recons_loss + feature_loss) + self.alpha * kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = paddle.randn(num_samples,
                         self.latent_dim)
        # z = torch.randn(num_samples,
        #                 self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


def get_model():
    return DFCVAE(3, 128), 'load model success'


if __name__ == "__main__":
    model, msg = get_model()
    print(msg)
    print(model)
    x = paddle.normal(shape=[144, 3, 64, 64])
    y = paddle.normal(shape=[144, 3, 64, 64])
    out = model(x)
    model.loss_function(out)
    print(len(out))
    # model.summary()
