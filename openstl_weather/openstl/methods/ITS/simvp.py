import torch
# from openstl.models.ITS.med_ciatt import SimVP_Model
from openstl.models.ITS.med_ciatt_mutinums import SimVP_Model


# from openstl.models.ITS.med_ciatt_1 import SimVP_Model
# from openstl.models.ITS.med_ciatt_dec1 import SimVP_Model
# from openstl.models.ITS.med_ciatt_lastdec1 import SimVP_Model
# from openstl.models.ITS.med_ciatt_encdown import SimVP_Model
# from openstl.models.ITS.med_predformer import SimVP_Model
# from openstl.models.ITS.med_ciatt_3dencdec import SimVP_Model
# from openstl.models.ITS.med_midsupertoken import SimVP_Model

# HIGH RE
# from openstl.models.ITS.med_ciatt_high import SimVP_Model
# from openstl.models.ITS.med_ciatt_high_superToken import SimVP_Model
# from openstl.models.ITS.med_ciatt_high_tokenmixer_ffnpool import SimVP_Model

# from openstl.models.ITS.med_ciatt_64_128 import SimVP_Model

# 消融
# from openstl.models.ITS.ablation.med_tau import SimVP_Model
# from openstl.models.ITS.ablation.med_ciatt_tau import SimVP_Model

# from openstl.models.ITS.ablation.twoStage1 import SimVP_Model

from openstl.methods.base_method import Base_method
import numpy as np
from openstl.utils.main_utils import print_log
from openstl.utils import print_log, check_dir
from openstl.core import get_optim_scheduler, timm_schedulers
from openstl.core import metric
import os.path as osp
from openstl.core.metrics import MAE,MSE,RMSE

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
        self.test_num = 0

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
        self.test_num  = self.test_num + 1
        batch_x, batch_y = batch
        # self.cout_cka(batch_x)
        pred_y, loss, loss_rec, loss_latent, loss_pre  = self(batch_x, batch_y)
        # pred_y = self.model.sample(batch_x)
        outputs = {'inputs': batch_x.cpu().numpy(), 'preds': pred_y.cpu().numpy(), 'trues': batch_y.cpu().numpy()}
        self.test_outputs.append(outputs)
        return outputs
    
    def on_test_epoch_end(self):
        print(f"-----------------------------------开始计算指标------------------------")

        results_all = {}
        # for k in self.test_outputs[0].keys():
        #     results_all[k] = np.concatenate([batch[k] for batch in self.test_outputs], axis=0)

        cnums = len(self.channel_names)
        mse_list = [0.]*cnums
        mae_list = [0.]*cnums
        num_batches = len(self.test_outputs)  # 获取批次数量
        rmse_list = [0.]*cnums
        for batch in self.test_outputs:
            pred = batch['preds']
            true = batch['trues']
            if self.hparams.test_mean is not None and self.hparams.test_std is not None:
                pred = pred * self.hparams.test_std + self.hparams.test_mean
                true = true * self.hparams.test_std + self.hparams.test_mean
            
            for i in range(cnums):
                mse = MSE(pred[:,:,i:i+1,...],true[:,:,i:i+1,...],self.spatial_norm)
                mse_list[i] = mse_list[i] + mse

                mae = MAE(pred[:,:,i:i+1,...],true[:,:,i:i+1,...],self.spatial_norm)
                mae_list[i] = mae_list[i] + mae

                rmse = RMSE(pred[:,:,i:i+1,...],true[:,:,i:i+1,...],self.spatial_norm)
                rmse_list[i] = rmse_list[i] + rmse

        # 计算平均值
        mse_list = [mse / num_batches for mse in mse_list]
        mae_list = [mae / num_batches for mae in mae_list]
        rmse_list = [rmse / num_batches for rmse in rmse_list]
        eval_log = ''
        for i in range(cnums):
            eval_log = eval_log + f'mse_{self.channel_names[i]}:{mse_list[i]}, '
            eval_log = eval_log + f'mae_{self.channel_names[i]}:{mae_list[i]}, '
            eval_log = eval_log + f'rmse{self.channel_names[i]}:{rmse_list[i]}, '
        
        # eval_res, eval_log = metric(results_all['preds'], results_all['trues'],
        #     self.hparams.test_mean, self.hparams.test_std, metrics=self.metric_list, 
        #     channel_names=self.channel_names, spatial_norm=self.spatial_norm,
        #     threshold=self.hparams.get('metric_threshold', None))
        
        # results_all['metrics'] = np.array([eval_res['mae'], eval_res['mse']])

        if self.trainer.is_global_zero:
            print_log(eval_log)
            folder_path = check_dir(osp.join(self.hparams.save_dir, 'saved'))

            # for np_data in ['metrics', 'inputs', 'trues', 'preds']:
            #     np.save(osp.join(folder_path, np_data + '.npy'), results_all[np_data])
        return results_all
    
    # 中心化 Gram 矩阵函数
    def center_gram_matrix(self, K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    # CKA 计算函数
    def linear_CKA(self, X, Y):
        # Reshape 4D 张量为 2D 张量: [batch_size, feature_dim]
        X = X.reshape(X.shape[0], -1)  # 展平
        Y = Y.reshape(Y.shape[0], -1)

        # 计算 Gram 矩阵
        K_X = X @ X.T
        K_Y = Y @ Y.T

        # 中心化
        K_X_centered = self.center_gram_matrix(K_X)
        K_Y_centered = self.center_gram_matrix(K_Y)

        # 计算 CKA
        numerator = np.trace(K_X_centered @ K_Y_centered)
        denominator = np.sqrt(np.trace(K_X_centered @ K_X_centered) * np.trace(K_Y_centered @ K_Y_centered))
        return numerator / denominator

     # Hook 函数，用于提取每层的输出
    def get_layer_output_hook(self, layer_outputs):
        def hook(module, input, output):
            # print(f"Captured layer output shape: {output.shape}")  # 调试输出每层输出的 shape
            layer_outputs.append(output.cpu().detach().numpy())
        return hook

    def cout_cka(self, batch_x):
        B, T, C, H, W = batch_x.shape
        x = batch_x.clone()
        x = x.view(B*T, C, H, W)

        # 获取模型的 encoders
        enc1 = self.model.enc_u10_q
        enc2 = self.model.enc_v10_q
        enc3 = self.model.enc_t2m_q
        enc4 = self.model.enc_tcc_q

        # Hook 存储输出的地方
        layer_outputs1 = []
        layer_outputs2 = []
        layer_outputs3 = []
        layer_outputs4 = []

        # 注册 forward hook 到 enc1, enc2, enc3, enc4 的每一层
        handles1 = []
        handles2 = []
        handles3 = []
        handles4 = []

        # 使用 modules() 遍历所有子层，捕获每个 BasicConv2d 层的输出
        for layer in enc1.modules():
            if isinstance(layer, torch.nn.Conv2d):  # 只捕获卷积层的输出
                handle = layer.register_forward_hook(self.get_layer_output_hook(layer_outputs1))
                handles1.append(handle)

        for layer in enc2.modules():
            if isinstance(layer, torch.nn.Conv2d):  # 只捕获卷积层的输出
                handle = layer.register_forward_hook(self.get_layer_output_hook(layer_outputs2))
                handles2.append(handle)

        for layer in enc3.modules():
            if isinstance(layer, torch.nn.Conv2d):  # 只捕获卷积层的输出
                handle = layer.register_forward_hook(self.get_layer_output_hook(layer_outputs3))
                handles3.append(handle)

        for layer in enc4.modules():
            if isinstance(layer, torch.nn.Conv2d):  # 只捕获卷积层的输出
                handle = layer.register_forward_hook(self.get_layer_output_hook(layer_outputs4))
                handles4.append(handle)

        # 前向传播，触发 hook 并获取每层的输出
        input1 = x[:, 0:2, :, :]  # 对应 enc1 的输入
        input2 = x[:, 2:3, :, :]  # 对应 enc2 的输入
        input3 = x[:, 3:4, :, :]  # 对应 enc3 的输入
        input4 = x[:, 4:5, :, :]  # 对应 enc4 的输入

        print("Starting forward pass...")
        enc1(input1)
        enc2(input2)
        enc3(input3)
        enc4(input4)
        print("Forward pass completed.")

        # 检查是否捕获到了每层的输出
        if not layer_outputs1:
            print("Warning: No output captured from enc1!")
        if not layer_outputs2:
            print("Warning: No output captured from enc2!")
        if not layer_outputs3:
            print("Warning: No output captured from enc3!")
        if not layer_outputs4:
            print("Warning: No output captured from enc4!")

        # 计算每层的 CKA
        cka_results = {}
        for i, (layer_output1, layer_output2, layer_output3, layer_output4) in enumerate(zip(layer_outputs1, layer_outputs2, layer_outputs3, layer_outputs4)):
            cka_12 = self.linear_CKA(layer_output1, layer_output2)
            cka_13 = self.linear_CKA(layer_output1, layer_output3)
            cka_14 = self.linear_CKA(layer_output1, layer_output4)
            cka_23 = self.linear_CKA(layer_output2, layer_output3)
            cka_24 = self.linear_CKA(layer_output2, layer_output4)
            cka_34 = self.linear_CKA(layer_output3, layer_output4)
            
            cka_results[f'Layer {i+1}'] = {
                'enc1-enc2': cka_12,
                'enc1-enc3': cka_13,
                'enc1-enc4': cka_14,
                'enc2-enc3': cka_23,
                'enc2-enc4': cka_24,
                'enc3-enc4': cka_34,
            }

        # 保存 CKA 结果字典为 .npy 文件
        np.save('/storage/linhaitao/lsh/openstl_weather/fenxi_result/enc_cka/'+f'cka_results_dict_{self.test_num}_.npy', cka_results)

        # 输出每层的 CKA 结果
        for layer, cka_values in cka_results.items():
            print(f"{layer} CKA:")
            print(f"enc1 和 enc2 CKA: {cka_values['enc1-enc2']}")
            print(f"enc1 和 enc3 CKA: {cka_values['enc1-enc3']}")
            print(f"enc1 和 enc4 CKA: {cka_values['enc1-enc4']}")
            print(f"enc2 和 enc3 CKA: {cka_values['enc2-enc3']}")
            print(f"enc2 和 enc4 CKA: {cka_values['enc2-enc4']}")
            print(f"enc3 和 enc4 CKA: {cka_values['enc3-enc4']}")

        # 移除 hooks
        for handle in handles1 + handles2 + handles3 + handles4:
            handle.remove()