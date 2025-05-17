import torch
from openstl.models.simvp_ema.simvp_prepost import SimVP_Model
from openstl.methods.base_method import Base_method


class SimVP(Base_method):
    r"""SimVP

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, **args):
        super().__init__(**args)

    def _build_model(self, **args):
        return SimVP_Model(**args)

    def forward(self, batch_x, batch_y=None, **kwargs):
        
        pre_y, loss, loss_rec, loss_latent, loss_pre = self.model(batch_x,batch_y)
        
        return pre_y, loss, loss_rec, loss_latent, loss_pre
    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pre_y, loss, loss_rec, loss_latent, loss_pre = self(batch_x,batch_y)
        # loss = self.criterion(pred_y, batch_y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pre_y, loss, loss_rec, loss_latent, loss_pre = self(batch_x,batch_y)
        # loss = self.criterion(pred_y, batch_y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('rec_loss', loss_rec, on_step=True, on_epoch=True, prog_bar=False)
        self.log('latent_loss', loss_latent, on_step=True, on_epoch=True, prog_bar=False)
        self.log('pre_loss', loss_pre, on_step=True, on_epoch=True, prog_bar=False)
        return loss
    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y, loss, loss_rec, loss_latent, loss_pre  = self(batch_x, batch_y)
        outputs = {'inputs': batch_x.cpu().numpy(), 'preds': pred_y.cpu().numpy(), 'trues': batch_y.cpu().numpy()}
        self.test_outputs.append(outputs)
        return outputs