from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import tqdm
from utils.pmae import qpmae, epmae, lepmae, alepmae, aqpmae
warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # Our modification on the new loss function
        if self.args.loss == "QPMAE":
            criterion = qpmae(self.args.alpha)
            print('Quadratic-Penalty MAE used')
        elif self.args.loss == "EPMAE":
            criterion = epmae
            print('Exponential-Penalty MAE used')
        elif self.args.loss == "LEPMAE":
            criterion = lepmae
            print('Linear-Exponential-Penalty MAE used')
        elif self.args.loss == "ALEPMAE":
            criterion = alepmae
            print('Adjusted Linear-Exponential-Penalty MAE used') 
        elif self.args.loss == "AQPMAE":
            criterion = aqpmae(self.args.alpha)
            print('Adjusted Quadratic-Penalty MAE used') 
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        torch.autograd.set_detect_anomaly(True)
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                if self.args.loss == 'QPMAE' or self.args.loss == 'EPMAE' or self.args.loss == 'LEPMAE' or self.args.loss == 'ALEPMAE' or self.args.loss == 'AQPMAE':
                    if pred[pred==0].nelement() == True:
                        print('Validating: prediction value 0 detected, change value to -1e-40')
                    # set the predicted 0s to a significantly small number
                    pred[pred==0] = -1e-40
                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
            
        torch.autograd.set_detect_anomaly(True)
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        print("-"*30)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader) # num of slides
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            ######################## progression bar ##########################
            print(('\n' + '%10s' * 3) % ('epoch', 'loss', 'gpu'))
            progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
            ###################################################################

            # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in progress_bar:
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float() # zeros for prediction
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        if self.args.loss == 'QPMAE' or self.args.loss == 'EPMAE' or self.args.loss == 'LEPMAE' or self.args.loss == 'ALEPMAE' or self.args.loss == 'AQPMAE':
                            if outputs[outputs==0].nelement() == True:
                                print('Training: prediction value 0 detected, change value to -1e-40')
                            # set the predicted 0s to a significantly small number
                            outputs[outputs==0] = -1e-40
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else: # we get here
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    # check if return data is being trained
                    if self.args.loss == 'QPMAE' or self.args.loss == 'EPMAE' or self.args.loss == 'LEPMAE' or self.args.loss == 'ALEPMAE' or self.args.loss == 'AQPMAE':
                        if outputs[outputs==0].nelement() == True:
                            print('Training: prediction value 0 detected')
                            # set the predicted 0s to a significantly small number
                            # 1e-50 would not pass the first multiplication ('MulBackward0' nan)
                            # 1e-45 would not pass the later torch.as_stride ('AsStridedBackward0' nan)
                            # 1e-40 would be just fine and close enough to 0
                            # outputs[outputs==0] = -1e-40
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

#                 if (i + 1) % 100 == 0:
#                     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
#                     speed = (time.time() - time_now) / iter_count
#                     left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
#                     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
#                     iter_count = 0
#                     time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                ######################## progression bar ##########################
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
                s = ('%10s' + '%10.4g' + '%10s') % ('%g/%g' % (epoch + 1, self.args.train_epochs), sum(train_loss) / (i + 1), mem)
                progress_bar.set_description(s)
                ###################################################################
                
            # original progress notice
            print("iters: {0} | loss: {1:.7f}".format(i + 1, loss.item()), end="")
            speed = (time.time() - time_now) / iter_count
            left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
            print('; speed: {:.4f}s/iter | left time: {:.4f}s'.format(speed, left_time), end="")
            iter_count = 0
            time_now = time.time()

            print("; cost time: {}".format(time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, result_file, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds, preds_inv = [], []
        trues, trues_inv = [], []
        folder_path = './RESULTS/' + result_file + '/test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                # inv-stand
                true_inv = test_data.inverse_transform(true.squeeze())
                pred_inv = test_data.inverse_transform(pred.squeeze())

                # stand for calculation
                preds.append(pred)
                trues.append(true)
                # inv-stand, for output
                preds_inv.append(pred_inv)
                trues_inv.append(true_inv)

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # stand
        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        print('test shape:', preds.shape, trues.shape)
        # inv-stand
        preds_inv = np.array(preds_inv)
        trues_inv = np.array(trues_inv)
        preds_inv = preds_inv.reshape(-1, preds_inv.shape[-2], preds_inv.shape[-1])
        trues_inv = trues_inv.reshape(-1, trues_inv.shape[-2], trues_inv.shape[-1])
        
        # result save
        folder_path = './RESULTS/' +  result_file + '/results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


        # check if the data type is return, if so include a penalized-MAE
        if self.args.is_return == True:
            mae, mse, rmse, mape, mspe, epmae, qpmae, lepmae, alepmae, aqpmae = metric(preds, trues, True, self.args.alpha, 2)
            print('mse:{}, mae:{}, epmae:{}, qpmae:{}, lepmae:{}, alepmae:{}, aqpmae:{}'.format(mae, mse, rmse, mape, mspe, epmae, qpmae, lepmae, alepmae, aqpmae)) # stand. for calculation
            f = open("./RESULTS/"+result_file+"/result_long_term_forecast.txt", 'a')
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}, epmae:{}, qpmae:{}, lepmae:{}, alepmae:{}, aqpmae:{}'.format(mae, mse, rmse, mape, mspe, epmae, qpmae, lepmae, alepmae, aqpmae))
            f.write('\n')
            f.write('\n')
            f.close()

            np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, epmae, qpmae, lepmae, alepmae, aqpmae]))
            np.save(folder_path + 'pred.npy', preds_inv) # inv-stand. for saving
            np.save(folder_path + 'true.npy', trues_inv)

        else: # not include the pmae
            mae, mse, rmse, mape, mspe = metric(preds, trues, False)
            print('mse:{}, mae:{}'.format(mse, mae)) # stand. for calculation
            f = open("./RESULTS/"+result_file+"/result_long_term_forecast.txt", 'a')
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}'.format(mse, mae))
            f.write('\n')
            f.write('\n')
            f.close()

            np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path + 'pred.npy', preds_inv) # inv-stand. for saving
            np.save(folder_path + 'true.npy', trues_inv)

        return
