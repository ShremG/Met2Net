import numpy as np
import torch.nn as nn
import os.path as osp
import lightning as l
from openstl.utils import print_log, check_dir
from openstl.core import get_optim_scheduler, timm_schedulers
from openstl.core import metric
from openstl.core.metrics import MAE,MSE,RMSE

class Base_method(l.LightningModule):

    def __init__(self, **args):
        super().__init__()

        if 'weather' in args['dataname']:
            self.metric_list, self.spatial_norm = args['metrics'], True
            self.channel_names = args['data_name'] if 'mv' in args['dataname'] else None
        else:
            self.metric_list, self.spatial_norm, self.channel_names = args['metrics'], False, None

        self.save_hyperparameters()
        self.model = self._build_model(**args)
        self.criterion = nn.MSELoss()
        self.test_outputs = []

    def _build_model(self):
        raise NotImplementedError
    
    def configure_optimizers(self):
        optimizer, scheduler, by_epoch = get_optim_scheduler(
            self.hparams, 
            self.hparams.epoch, 
            self.model, 
            self.hparams.steps_per_epoch
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "epoch" if by_epoch else "step"
            },
        }
    
    def lr_scheduler_step(self, scheduler, metric):
        if any(isinstance(scheduler, sch) for sch in timm_schedulers):
            scheduler.step(epoch=self.current_epoch)
        else:
            if metric is None:
                scheduler.step()
            else:
                scheduler.step(metric)

    def forward(self, batch):
        NotImplementedError
    
    def training_step(self, batch, batch_idx):
        NotImplementedError

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        loss = self.criterion(pred_y, batch_y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss
    
    
    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        outputs = {'inputs': batch_x.cpu().numpy(), 'preds': pred_y.cpu().numpy(), 'trues': batch_y.cpu().numpy()}
        self.test_outputs.append(outputs)
        # print('-------')
        return outputs
    
    def on_test_epoch_end(self):
        print(f"-----------------------------------开始计算指标------------------------")
        results_all = {}
        # for k in self.test_outputs[0].keys():
        #     results_all[k] = np.concatenate([batch[k] for batch in self.test_outputs], axis=0)

        cnums = len(self.channel_names) if self.channel_names != None else 1
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
            eval_log = eval_log + f"mse_{self.channel_names[i] if self.channel_names != None else ''}:{mse_list[i]}, "
            eval_log = eval_log + f"mae_{self.channel_names[i] if self.channel_names != None else ''}:{mae_list[i]}, "
            eval_log = eval_log + f"rmse{self.channel_names[i] if self.channel_names != None else ''}:{rmse_list[i]}, "
        
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

    # def on_test_epoch_end(self):
    #     # print(f"-----------------------------------开始计算指标------------------------")
    #     results_all = {}
    #     for k in self.test_outputs[0].keys():
    #         results_all[k] = np.concatenate([batch[k] for batch in self.test_outputs], axis=0)
        
    #     # print(f"开始计算指标,trues:{results_all['trues'].shape}, preds:{results_all['preds'].shape}")
    #     eval_res, eval_log = metric(results_all['preds'], results_all['trues'],
    #         self.hparams.test_mean, self.hparams.test_std, metrics=self.metric_list, 
    #         channel_names=self.channel_names, spatial_norm=self.spatial_norm,
    #         threshold=self.hparams.get('metric_threshold', None))
        
    #     results_all['metrics'] = np.array([eval_res['mae'], eval_res['mse']])

    #     if self.trainer.is_global_zero:
    #         print_log(eval_log)
    #         folder_path = check_dir(osp.join(self.hparams.save_dir, 'saved'))

    #         for np_data in ['metrics', 'inputs', 'trues', 'preds']:
    #             np.save(osp.join(folder_path, np_data + '.npy'), results_all[np_data])
    #     return results_all