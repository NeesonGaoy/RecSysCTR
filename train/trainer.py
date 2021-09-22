'''
Author: sysu.gaoyong
Email: ygaoneeson@gmail.com
Date: 2021-09-22 11:18:09
LastEditTime: 2021-09-22 17:55:23
LastEditors: sysu.gaoyong
FilePath: /RecSysCTR/train/trainer.py
Copyright (c) 2011 Neeson.GaoYong All rights reserved.
'''
# coding:utf-8
import os
import json

import numpy as np
import torch
from train.utils import get_model, get_optimizer, get_log


def get_metric(labels, probs, sig_num=4):
    auc = roc_auc_score(labels, probs)
    loss = log_loss(labels, probs)
    auc = round(auc, sig_num)
    loss = round(loss, sig_num)

    CTR = np.sum(labels) / len(labels.squeeze().tolist())
    avg_pCTR = np.mean(probs)
    ratio = (avg_pCTR - CTR) / CTR
    
    CTR = round(CTR, sig_num)
    avg_pCTR = round(avg_pCTR.item(), sig_num)
    ratio = round(ratio, sig_num)
    # print("avg_pCTR.type: {} ".format( type(avg_pCTR) ))
    # print("avg_pCTR: {} ".format( avg_pCTR ))
    return auc, loss, CTR, avg_pCTR, ratio

class Trainer(object):
    def __init__(self, data_loader,
                       data_config, 
                       model_config, 
                       train_config,
                       use_cuda=False):
        """
        data_config: {
            "field_num": field_num,
            "id_vocab_size": ID_size,
        }

        model_config: {
            "model_name": , ["deepfm", "dcn"]
            "emb_dim": ,
            "emb_type": ,  ["normal", "quant"]
            "mlp_dims": ,
            "dropout": ,
            "num_cross_layer": ,

            "quant_config": {
                "quant_method": ,
                "weight_bits": ,
                "layerwise": ,
                "learnable": ,
                "symmetric": ,
                "clip_val": ,
            },
        }
        train_config: {
            "batch_size": ,
            "print_steps": , 
            "eval_steps": ,
            "early_stop_steps": ,

            "l2_sparse": ,
            "l2_dense": ,

            "optimizer": , ["sgd", "adam", "rmsprop"]
            "sgd_lr": ,
            "adam_lr": ,
            "rmsp_lr": ,
            "rmsp_alpha": ,
            "rmsp_momentum": ,

            "log_path": ,
            "log_name": ,

            "load_pretrain": ,
            "pretrain_path": ,
            "pretrain_model_name": ,
        }
        
        """
        self.data_loader = data_loader

        self.data_config = data_config
        self.model_config = model_config
        self.train_config = train_config
        self.use_cuda = use_cuda

        self.print_steps = train_config["print_steps"]
        self.eval_steps = train_config["eval_steps"]
        self.early_stop_steps = train_config["early_stop_steps"]

        self.init_model_args()
        self.load_pretrain()

    def init_model_args(self):
        self.model = get_model(self.model_config["model_name"], 
                               self.data_config, 
                               self.model_config)
        self.optimizer = get_optimizer(self.model, 
                                       self.train_config)
        
        # self.criterion = torch.nn.BCELoss(reduction='mean')
        self.criterion = torch.nn.functional.binary_cross_entropy_with_logits

        self.logger = get_log(self.train_config["log_path"], self.train_config["log_name"])
        # self.logger.info("data_config: {} ".format(self.data_config))
        # self.logger.info("model_config: {} ".format(self.model_config))
        # self.logger.info("train_config: {} ".format(self.train_config))

        self.logger.info("data_config: {} ".format(json.dumps(self.data_config, indent=2) ))
        self.logger.info("model_config: {} ".format(json.dumps(self.model_config, indent=2) ))
        self.logger.info("train_config: {} ".format(json.dumps(self.train_config, indent=2) ))
        
        self.save_model_path = os.path.join(self.train_config["log_path"], self.train_config["log_name"])
        os.makedirs(self.save_model_path, exist_ok=True)
        self.save_model_name = "saved_step{}_model.pt"

        self.batch_size = self.train_config["batch_size"]

    def load_pretrain(self):
        if self.train_config["load_pretrain"]:
            pretrain_model_path = os.path.join(self.train_config["pretrain_path"], 
                                               self.train_config["pretrain_model_name"])
            pretrain_dict = torch.load(pretrain_model_path)
            model_dict = self.model.state_dict()
            pretrain_dict = {k: v for k,v in pretrain_dict.items() if k in model_dict}
            model_dict.update(pretrain_dict)
            self.model.load_state_dict(model_dict)
    
    def __update(self, inputs):
        self.model.train()
        self.optimizer.zero_grad()
        if self.use_cuda:
            inputs = {k : v.cuda() for k,v in inputs.items()}
        logits, out_dict = self.model.forward(inputs["ids"])
        # prob = torch.sigmoid(logits)
        # print("logits: {} ".format( logits.cpu().detach().numpy().tolist()[:2] ))
        # print("label: {} ".format( inputs["label"].cpu().detach().numpy().tolist()[:2] ))
        loss = self.criterion(logits, inputs["label"].squeeze())
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def __evaluate(self, data_type, eval_batch_num=None, save_pred=False):
        self.model.eval()
        prob_list, label_list = [], []
        test_data_iter = self.data_loader.get_data(data_type, batch_size=self.batch_size, epoch=1)
        for step,inputs in test_data_iter:
            if self.use_cuda:
                inputs = {k : v.cuda() for k,v in inputs.items()}
            logit, out_dict = self.model.forward(inputs["ids"])
            prob = torch.sigmoid(logit).detach().cpu().numpy()
            label = inputs["label"].detach().cpu().numpy()
            prob_list.extend( prob.squeeze() )
            label_list.extend( label.squeeze() )
            if eval_batch_num is not None:
                if step >= eval_batch_num:
                    break
        probs = np.asarray(prob_list)
        labels = np.asarray(label_list)
        if save_pred:
            np.save(os.path.join(self.save_model_path, "probs.npy"), probs)
            np.save(os.path.join(self.save_model_path, "labels.npy"), labels)
        # 
        auc, loss, CTR, avg_pCTR, ratio = get_metric(labels, probs)
        return auc, loss, CTR, avg_pCTR, ratio
    
    def train(self, max_epoch):
        best_auc = 0.0
        best_metric = []
        best_epoch = 0
        best_steps = 0
        stop_flag = False
        step = 1
        for epoch in range(1, max_epoch+1, 1):
            train_data_iter = self.data_loader.get_data("train", batch_size=self.batch_size, epoch=1)
            for i,inputs in train_data_iter:
                step += 1
                loss = self.__update(inputs)
                if step % self.print_steps == 0:
                    self.logger.info("Epoch:{}; steps:{}; Train Loss: {} ".format(epoch, step, loss))
                if step % self.eval_steps == 0:
                    v_auc, v_loss, v_CTR, v_pCTR, v_ratio = self.__evaluate(data_type="val", eval_batch_num=None, save_pred=False)
                    self.logger.info("VAL  Epoch:{}; steps:{}; AUC:{:.6f}; loss:{:.6f}; CTR:{:.6f}; pCTR:{:.6f}; ratio:{:.6f} ".format(
                                              epoch, step,       v_auc,       v_loss,        v_CTR,     v_pCTR,      v_ratio ))
                    if v_auc - best_auc > 1e-7:
                        best_auc = v_auc
                        best_metric = [v_auc, v_loss, v_CTR, v_pCTR, v_ratio]
                        best_epoch = epoch
                        best_steps = step
                        t_auc, t_loss, t_CTR, t_pCTR, t_ratio = self.__evaluate(data_type="test", eval_batch_num=None, save_pred=True)
                        self.logger.info("TEST Epoch:{}; steps:{}; AUC:{:.6f}; loss:{:.6f}; CTR:{:.6f}; pCTR:{:.6f}; ratio:{:.6f} ".format(
                                                  epoch,    step,    t_auc,       t_loss,      t_CTR,      t_pCTR,      t_ratio ))
                        torch.save(self.model.state_dict(), os.path.join(self.save_model_path, self.save_model_name.format(best_steps)))
                    if step - best_steps >= self.early_stop_steps:
                        stop_flag = True
                        self.logger.info("Best Epoch:{}; step:{}; VAL AUC:{:.6f}; loss:{:.6f}; CTR:{:.6f}; pCTR:{:.6f}; ratio:{:.6f} ".format( 
                                                best_epoch, best_steps, *best_metric))
            if stop_flag:
                break

class Trainer_Distillation(object):
    def __init__(self, data_loader,
                       data_config, 
                       model_config, 
                       train_config,
                       use_cuda=False):
        """
        data_config: {
            "field_num": field_num,
            "id_vocab_size": ID_size,
        }

        model_config: {
            "model_name": , ["deepfm", "dcn"]
            "emb_dim": ,
            "emb_type": ,  ["normal", "quant"]
            "mlp_dims": ,
            "dropout": ,
            "num_cross_layer": ,

            "quant_config": {
                "quant_method": ,
                "weight_bits": ,
                "layerwise": ,
                "learnable": ,
                "symmetric": ,
                "clip_val": ,
            },
        }
        train_config: {
            "batch_size": ,
            "print_steps": , 
            "eval_steps": ,
            "early_stop_steps": ,

            "l2_sparse": ,
            "l2_dense": ,

            "opt_name": , ["sgd", "adam", "rmsprop"]
            "sgd_lr": ,
            "adam_lr": ,
            "rmsp_lr": ,
            "rmsp_alpha": ,
            "rmsp_momentum": ,

            "log_path": ,
            "log_name": ,

            "load_pretrain": ,
            "pretrain_path": ,
            "pretrain_model_name": ,
        }
        
        """
        self.data_loader = data_loader

        self.data_config = data_config
        self.model_config = model_config
        self.train_config = train_config
        self.use_cuda = use_cuda
        # 
        self.print_steps = train_config["print_steps"]
        self.eval_steps = train_config["eval_steps"]
        self.early_stop_steps = train_config["early_stop_steps"]
        # 
        self.mse_coef = self.train_config["mse_coef"]

        self.init_model_args()
        self.load_pretrain()

    def init_model_args(self):
        model_config["emb_type"] = "normal" # teacher model. 
        self.teacher_model = get_model(self.model_config["model_name"], 
                                       self.data_config, 
                                       self.model_config)
        model_config["emb_type"] = "quant"  # student model. 
        self.student_model = get_model(self.model_config["model_name"], 
                                       self.data_config, 
                                       self.model_config)
        
        self.teacher_opt = get_optimizer(self.teacher_model, self.training_config)
        self.student_opt = get_optimizer(self.student_model, self.training_config)

        # self.criterion = torch.nn.BCELoss(reduction='mean')
        self.criterion = torch.nn.functional.binary_cross_entropy_with_logits

        self.mse_loss  = torch.nn.MSELoss(reduction=True)

        self.logger = get_log(self.train_config["log_path"], self.train_config["log_name"])

        self.save_model_path = os.path.join(self.train_config["log_path"], self.train_config["log_name"])
        os.makedirs(self.save_model_path, exist_ok=True)
        self.save_t_model_name = "saved_step{}_teacher_model.pt"
        self.save_s_model_name = "saved_step{}_student_model.pt"

        self.batch_size = self.train_config["batch_size"]


    def load_pretrain(self):
        if self.train_config["load_pretrain"]:
            pretrain_model_path = os.path.join(self.train_config["pretrain_path"], 
                                               self.train_config["pretrain_model_name"])
            pretrain_dict = torch.load(pretrain_model_path)

            t_model_dict = self.teacher_model.state_dict()
            t_pretrain_dict = {k: v for k,v in pretrain_dict.items() if k in t_model_dict}
            t_model_dict.update(t_pretrain_dict)
            self.teacher_model.load_state_dict(t_model_dict)

            s_model_dict = self.student_model.state_dict()
            s_pretrain_dict = {k: v for k,v in pretrain_dict.items() if k in s_model_dict}
            s_model_dict.update(s_pretrain_dict)
            self.student_model.load_state_dict(s_model_dict)
    
    def __update(self, inputs):
        self.teacher_model.train()
        self.student_model.train()

        if self.use_cuda:
            inputs = {k : v.cuda() for k,v in inputs.items()}
        # 
        t_logits, t_out_dict = self.teacher_model.forward(inputs["ids"])
        s_logits, s_out_dict = self.student_model.forward(inputs["ids"])
        # t_prob = torch.sigmoid(t_logits)
        # s_prob = torch.sigmoid(s_logits)
        t_logloss = self.criterion(t_logits, inputs["label"].squeeze())
        s_logloss = self.criterion(s_logits, inputs["label"].squeeze())
        distillation_loss_list = [self.mse_loss(t_out_dict[key], s_out_dict[key]) \
                                    for key in s_out_dict.keys()]
        distillation_loss = self.mse_coef * np.sum(distillation_loss_list)

        if self.train_config["teacher_update"]:
            self.teacher_opt.zero_grad()
            self.student_opt.zero_grad()

            loss = t_logloss + s_logloss + distillation_loss
            loss.backward()

            self.teacher_opt.step()
            self.student_opt.step()
        else:
            self.student_opt.zero_grad()

            loss = t_logloss + distillation_loss
            loss.backward()

            self.student_opt.step()
        return loss.item()
    
    def __evaluate(self, data_type, eval_batch_num=None, save_pred=False):
        self.teacher_model.eval()
        self.student_model.eval()
        label_list, t_prob_list, s_prob_list = [], [], []
        data_iter = self.data_loader.get_data(data_type, batch_size=self.batch_size, epoch=1)
        step = 0
        for inputs in data_iter:
            step += 1
            if self.use_cuda:
                inputs = {k : v.cuda() for k,v in inputs.items()}
            t_prob = self.teacher_model.forward(inputs["ids"])
            s_prob = self.student_model.forward(inputs["ids"])
            label = inputs["label"].detach().cpu().numpy()
            t_prob = torch.sigmoid(t_prob).detach().cpu().numpy()
            s_prob = torch.sigmoid(s_prob).detach().cpu().numpy()
            label_list.append(label)
            t_prob_list.append(t_prob)
            s_prob_list.append(s_prob)
            if eval_batch_num is not None:
                if step >= eval_batch_num:
                    break
        labels = np.concatenate(label_list).astype(np.float32)
        t_probs = np.concatenate(t_prob_list).astype(np.float32)
        s_probs = np.concatenate(s_prob_list).astype(np.float32)
        if save_pred:
            np.save(os.path.join(self.save_model_path, "labels.npy"), labels)
            np.save(os.path.join(self.save_model_path, "teacher_probs.npy"), t_probs)
            np.save(os.path.join(self.save_model_path, "student_probs.npy"), s_probs)
        # t_auc, t_loss, CTR, t_pCTR, t_ratio = get_metric(labels, t_probs)
        s_auc, s_loss, CTR, s_pCTR, s_ratio = get_metric(labels, s_probs)
        # return (t_auc, t_loss, s_auc, s_loss), (CTR, t_pCTR, t_ratio, s_pCTR, s_ratio)
        return s_auc, s_loss, CTR, s_pCTR, s_ratio
    
    def train(self, max_epoch):
        best_auc = 0.0
        best_metric = []
        best_epoch = 0
        best_steps = 0
        stop_flag = False
        step = 1
        for epoch in range(max_epoch):
            train_data_iter = self.data_loader.get_batch_data("train", batch_size=self.batch_size, epoch=1)
            for inputs in train_data_iter:
                step += 1
                loss = self.__update(inputs)
                if step % self.print_steps == 0:
                    self.logger.info("Epoch:{}; steps:{}; Train Loss: {} ".format(epoch, step, loss))
                if step % self.eval_steps == 0:
                    v_auc, v_loss, v_CTR, v_pCTR, v_ratio = self.__evaluate("val", eval_batch_num=None, save_pred=False)
                    self.logger.info("VAL  Epoch:{}; steps:{}; AUC:{}; loss:{};  CTR:{}; pCTR:{}; ratio:{} ".format(
                                            epoch, steps,       v_auc, v_loss,     v_CTR, v_pCTR, v_ratio ))
                    if v_auc - best_auc > 1e-7:
                        best_auc = val_auc
                        best_metric = [v_auc, v_loss, v_CTR, v_pCTR, v_ratio]
                        best_epoch = epoch
                        best_steps = steps
                        t_auc, t_loss, t_CTR, t_pCTR, t_ratio = self.__evaluate("test", eval_batch_num=None, save_pred=True)
                        self.logger.info("TEST Epoch:{}; steps:{}; AUC:{}; loss:{};  CTR:{}; pCTR:{}; ratio:{} ".format(
                                                    epoch, steps,   t_auc, t_loss,    t_CTR, t_pCTR, t_ratio))
                        torch.save(self.teacher_model.state_dict(), os.path.join(self.save_model_path, 
                                                                        self.save_t_model_name.format(best_steps)))
                        torch.save(self.student_model.state_dict(), os.path.join(self.save_model_path, 
                                                                        self.save_s_model_name.format(best_steps)))
                        # 
                        self.logger.info("Current Best VAL  Epoch:{}; steps:{}; AUC:{}; loss:{};  CTR:{}; pCTR:{}; ratio:{} ".format(
                                                           best_epoch, best_steps,    *best_metric ))
                    if step - best_steps >= self.early_stop_steps:
                        stop_flag = True
                        self.logger.info("Best Epoch:{}; step:{};  VAL AUC:{}; loss:{};  CTR:{}; pCTR:{}; ratio:{} ".format( 
                                            best_epoch, best_steps, *best_metric))
            if stop_flag:
                break

