import torch
from openstl.models.simvp_ema.simvp import SimVP_Model
# from openstl.models.simvp_ema.simvp_oneencdec import SimVP_Model
# from openstl.models.simvp_ema.simvp_noLatent import SimVP_Model
# from openstl.models.simvp_ema.simvp_oneencdec_noLatent import SimVP_Model
# from openstl.models.simvp_ema.videogpt import SimVP_Model
# from openstl.models.simvp_ema.videogpt_oneencdec import SimVP_Model
# from openstl.models.simvp_ema.simvp_new import SimVP_Model
# from openstl.models.simvp_ema.simvp_new_3ddec import SimVP_Model
# from openstl.models.simvp_ema.simvp_new_3ddec_ci import SimVP_Model
# from openstl.models.simvp_ema.simvp_new_3ddec_3denc import SimVP_Model
# from openstl.models.simvp_ema.simvp_new_3ddec_ciatt import SimVP_Model
# from openstl.models.simvp_ema.simvp_new_3ddec_ciatt_2hid import SimVP_Model
# from openstl.models.simvp_ema.simvp_new_1enc1dec import SimVP_Model
# from openstl.models.simvp_ema.simvp_new_3ddec_ciatt_3level import SimVP_Model
# from openstl.models.simvp_ema.simvp_new_3ddec_ciatt_5c import SimVP_Model
# from openstl.models.simvp_ema.simvp_new_ciatt_3level import SimVP_Model
# from openstl.models.simvp_ema.simvp_new_ciatt_IE import SimVP_Model
# from openstl.models.simvp_ema.simvp_new_ciatt_med_3level import SimVP_Model
# from openstl.models.simvp_ema.simvp_new_ciatt_patch import SimVP_Model

# ablation
# from openstl.models.simvp_ema.ablation.no_3ddec import SimVP_Model

from openstl.methods.base_method import Base_method
import numpy as np
from openstl.utils.main_utils import print_log

class SimVP(Base_method):
    r"""SimVP

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_

    """

    def __init__(self, **args):
        super().__init__(**args)
        self.rec_loss_list = []
        self.latent_loss_list = []
        self.pre_loss_list = []
        self.w1 = 1.0
        self.w2 = 1.0
        self.w3 = 1.0
        self.flag_w = False

    def _build_model(self, **args):
        return SimVP_Model(**args)

    def forward(self, batch_x, batch_y=None, **kwargs):
        
        pre_y, loss, loss_rec, loss_latent, loss_pre = self.model(batch_x,batch_y)
        # pre_y, loss, loss_rec, loss_latent, loss_pre = self.model(batch_x)
        
        return pre_y, loss, loss_rec, loss_latent, loss_pre
    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pre_y, loss, loss_rec, loss_latent, loss_pre = self(batch_x,batch_y)
        self.rec_loss_list.append(loss_rec.item())
        self.latent_loss_list.append(loss_latent.item())
        self.pre_loss_list.append(loss_pre.item())
 
        train_loss = self.w1*loss_rec + self.w2*loss_latent + self.w3*loss_pre
        # loss = self.criterion(pred_y, batch_y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss # train_loss
    
    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pre_y, loss, loss_rec, loss_latent, loss_pre = self(batch_x,batch_y)
        
        val_loss = self.w1*loss_rec + self.w2*loss_latent + self.w3*loss_pre
        # loss = self.criterion(pred_y, batch_y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('rec_loss', loss_rec, on_step=True, on_epoch=True, prog_bar=False)
        self.log('latent_loss', loss_latent, on_step=True, on_epoch=True, prog_bar=False)
        self.log('pre_loss', loss_pre, on_step=True, on_epoch=True, prog_bar=False)
        return loss # val_loss
    
    def on_validation_epoch_end(self):

        if self.current_epoch==0 and self.flag_w:
            self.m1 = np.mean(self.rec_loss_list)
            self.m2 = np.mean(self.latent_loss_list)
            self.m3 = np.mean(self.pre_loss_list)
            self.w1 = 1. / self.m1 * 0.1
            self.w2 = 1. / self.m2 * 0.1 * 0.1
            self.w3 = 1. / self.m3 * 0.1 * 0.1 * 0.1
            print_log(f"w1 : {self.w1} | w2 : {self.w2} | w3 : {self.w3}")

        # if self.flag_w:
        #     if self.current_epoch < 0.3*self.hparams.epoch:
        #         self.w1 = 1. / self.m1 
        #         self.w2 = 1. / self.m2 * 0.1
        #         self.w3 = 1. / self.m3 * 0.1 * 0.1
        #         print_log(f"w1 : {self.w1} | w2 : {self.w2} | w3 : {self.w3}")
        #     else:
        #         self.w1 = 1. / self.m1 
        #         self.w2 = 1. / self.m2
        #         self.w3 = 1. / self.m3 * 0.1 * 0.1
        #         print_log(f"w1 : {self.w1} | w2 : {self.w2} | w3 : {self.w3}")
        
        self.flag_w = True

    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y, loss, loss_rec, loss_latent, loss_pre  = self(batch_x, batch_y)
        outputs = {'inputs': batch_x.cpu().numpy(), 'preds': pred_y.cpu().numpy(), 'trues': batch_y.cpu().numpy()}
        self.test_outputs.append(outputs)
        return outputs