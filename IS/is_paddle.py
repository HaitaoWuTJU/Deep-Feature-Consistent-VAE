



def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    N = len(imgs)
    assert batch_size > 0
    assert N > batch_size
    # if cuda:
    #     dtype = torch.cuda.FloatTensor
    # else:
    #     if torch.cuda.is_available():
    #         print("WARNING: You have a CUDA device, so you should probably set cuda=True")
    #     dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

inception_score(path='/app/wht/paddle/DeepFeatureConsistentVAE/paddle_dfcvae/imgGenerateResult/vae123_conv_his_loss_54300', cuda=True, batch_size=32, resize=True, splits=10)
print()