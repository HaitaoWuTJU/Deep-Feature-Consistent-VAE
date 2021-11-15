import paddle
def get_optimizers(models,config):
    optimizers = paddle.optimizer.Adam(parameters=models.parameters(),learning_rate=config['LR'],weight_decay=config['weight_decay'])
    return optimizers