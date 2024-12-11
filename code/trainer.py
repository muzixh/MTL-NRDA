import torch
import numpy as np
import pandas as pd
from torch.utils import data
from utils import load_data
from models import MixHopNetwork
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import os
import time

class ModelSaver:
    def __init__(self, args):
        self.args = args
        self.max_score = -np.inf
        self.no_improvement = 0
        self.weights = {'DLI': 0.3, 'DMI': 0.3, 'MLI': 0.4} 
        self.patience = args.early_stopping
        self.min_improvement = 0.001 
        self.epoch = 0

    def update_weights(self):

        if self.epoch < 16:
            self.weights = {'DLI': 0.3, 'DMI': 0.3, 'MLI': 0.4}
        elif 16 <= self.epoch < 35:
            self.weights = {'DLI': 0.3, 'DMI': 0.35, 'MLI': 0.35}
        else:
            self.weights = {'DLI': 1/3, 'DMI': 1/3, 'MLI': 1/3}

    def calculate_score(self, roc_val_DLI, roc_val_DMI, roc_val_MLI, f1_val_DLI, f1_val_DMI, f1_val_MLI):
        return (self.weights['DLI'] * (roc_val_DLI + f1_val_DLI) +
                self.weights['DMI'] * (roc_val_DMI + f1_val_DMI) +
                self.weights['MLI'] * (roc_val_MLI + f1_val_MLI))

    def should_save_model(self, roc_val_DLI, roc_val_DMI, roc_val_MLI, f1_val_DLI, f1_val_DMI, f1_val_MLI):
        self.epoch += 1
        self.update_weights()
        
        current_score = self.calculate_score(roc_val_DLI, roc_val_DMI, roc_val_MLI, f1_val_DLI, f1_val_DMI, f1_val_MLI)
        improvement = current_score - self.max_score

        if improvement > self.min_improvement:
            self.max_score = current_score
            self.no_improvement = 0
            return True
        else:
            self.no_improvement += 1
            return False

    def should_stop(self):
        return self.no_improvement >= self.patience

class Trainer(object):

    def __init__(self, args):
        self.args = args
        self.device = self.args.device
        self.setup_features()

        self.model = MixHopNetwork(self.args, self.feature_number, 1, self.device)

        self.model = self.model.to(self.device)
        print(self.model)

    def setup_features(self):

        self.propagation_matrix, self.features, self.idx_map, Data_class = load_data(self.args)

        train_params = {'batch_size': self.args.batch_size,
                        'shuffle': True,
                        'num_workers': 6,
                        'drop_last': True}

        test_params = {'batch_size': self.args.batch_size,
                       'shuffle': False,
                       'num_workers': 6}

        data_path = f"./data/{self.args.network_type}/fold{self.args.fold_id}"
        if self.args.ratio:
            data_path = f"./data/{self.args.network_type}/{self.args.train_percent}/fold{self.args.fold_id}"

        print(f"Data folder: {data_path}")
        df_train = pd.read_csv(data_path + '/train.csv')
        df_val_DLI = pd.read_csv(data_path + '/val_DLI.csv')
        df_val_DMI = pd.read_csv(data_path + '/val_DMI.csv')
        df_val_MLI = pd.read_csv(data_path + '/val_MLI.csv')
        df_test_DLI = pd.read_csv(data_path + '/test_DLI.csv')
        df_test_DMI = pd.read_csv(data_path + '/test_DMI.csv')
        df_test_MLI = pd.read_csv(data_path + '/test_MLI.csv')

        training_set = Data_class(self.idx_map, df_train.label.values, df_train)
        self.train_loader = data.DataLoader(training_set, **train_params)

        validation_set_DLI  = Data_class(self.idx_map, df_val_DLI.label.values, df_val_DLI)
        validation_set_DMI  = Data_class(self.idx_map, df_val_DMI.label.values, df_val_DMI)
        validation_set_MLI  = Data_class(self.idx_map, df_val_MLI.label.values, df_val_MLI)
        self.val_loader_DLI = data.DataLoader(validation_set_DLI, **test_params)
        self.val_loader_DMI = data.DataLoader(validation_set_DMI, **test_params)
        self.val_loader_MLI = data.DataLoader(validation_set_MLI, **test_params)

        test_set_DLI = Data_class(self.idx_map, df_test_DLI.label.values, df_test_DLI)
        test_set_DMI = Data_class(self.idx_map, df_test_DMI.label.values, df_test_DMI)
        test_set_MLI = Data_class(self.idx_map, df_test_MLI.label.values, df_test_MLI)
        self.test_loader_DLI = data.DataLoader(test_set_DLI, **test_params)
        self.test_loader_DMI = data.DataLoader(test_set_DMI, **test_params)
        self.test_loader_MLI = data.DataLoader(test_set_MLI, **test_params)

        self.feature_number = self.features["dimensions"][1]

        if self.args.ratio:
            self.model_save_folder = f"trained_models/{self.args.model}/network_{self.args.network_type}/{self.args.train_percent}/order_{len(self.args.layers_1)}/fold{self.args.fold_id}/"
        else:
            self.model_save_folder = f"trained_models/{self.args.model}/network_{self.args.network_type}/order_{len(self.args.layers_1)}/fold{self.args.fold_id}/"

        if not os.path.exists(self.model_save_folder):
            os.makedirs(self.model_save_folder)

    def fit(self):

        no_improvement = 0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        max_auc_DLI = 0
        max_auc_DMI = 0
        max_auc_MLI = 0
        total_loss_history = []
        model_saver = ModelSaver(self.args)

        t_total = time.time()
        print('Start Training...')
        for epoch in range(self.args.epochs):
            t = time.time()
            print('-------- Epoch ' + str(epoch + 1) + ' --------')
            # y_pred_train = []
            # y_label_train = []
            y_pred_disease_lncrna = []
            y_label_disease_lncrna = []
            y_pred_lncrna_mirna = []
            y_label_lncrna_mirna = []
            y_pred_mirna_disease = []
            y_label_mirna_disease = []


            epoch_loss = 0

            for i, (label, pairs) in enumerate(self.train_loader):
                self.model.train()
                self.optimizer.zero_grad()


                label = label.to(self.device)

                # prediction, latent_feat = self.model(self.propagation_matrix, self.features, pairs)
                prediction_DLI, prediction_DMI, prediction_MLI, latent_feat = self.model(self.propagation_matrix, self.features, pairs)

                loss_DLI = torch.nn.functional.binary_cross_entropy_with_logits(prediction_DLI.squeeze(), label.float())
                loss_DMI = torch.nn.functional.binary_cross_entropy_with_logits(prediction_DMI.squeeze(), label.float())
                loss_MLI = torch.nn.functional.binary_cross_entropy_with_logits(prediction_MLI.squeeze(), label.float())
                weight_DLI = 1.0
                weight_DMI = 1.0
                weight_MLI = 1.0
                total_loss = (weight_DLI * loss_DLI + 
                              weight_MLI * loss_MLI + 
                              weight_DMI * loss_DMI)

                total_loss.backward()

                self.optimizer.step()
                epoch_loss += total_loss.item()
                total_loss_history.append(total_loss)

                label_ids = label.to('cpu').numpy()

                y_label_disease_lncrna += label_ids.flatten().tolist()
                y_pred_disease_lncrna += prediction_DLI.flatten().tolist()
                y_label_lncrna_mirna += label_ids.flatten().tolist()
                y_pred_lncrna_mirna += prediction_MLI.flatten().tolist()
                y_label_mirna_disease += label_ids.flatten().tolist()
                y_pred_mirna_disease += prediction_DMI.flatten().tolist()

                if i % 100 == 0:
                    print(f'Epoch: {epoch + 1}/{self.args.epochs} Iteration: {i + 1}/{len(self.train_loader)}')
                    print(f'Total loss: {total_loss.item():.4f}')
                    print(f'Disease-lncRNA loss: {loss_DLI.item():.4f}')
                    print(f'miRNA-Disease loss: {loss_DMI.item():.4f}')
                    print(f'lncRNA-miRNA loss: {loss_MLI.item():.4f}')


                roc_train_DLI = roc_auc_score(y_label_disease_lncrna, y_pred_disease_lncrna)
                roc_train_MLI = roc_auc_score(y_label_lncrna_mirna, y_pred_lncrna_mirna)
                roc_train_DMI = roc_auc_score(y_label_mirna_disease, y_pred_mirna_disease)

            if not self.args.fastmode:
                preds_DLI, roc_val_DLI, prc_val_DLI, f1_val_DLI, loss_val_DLI = self.score(self.val_loader_DLI, 'DLI')
                preds_DMI, roc_val_DMI, prc_val_DMI, f1_val_DMI, loss_val_DMI = self.score(self.val_loader_DMI, 'DMI')
                preds_MLI, roc_val_MLI, prc_val_MLI, f1_val_MLI, loss_val_MLI = self.score(self.val_loader_MLI, 'MLI')
                
            if model_saver.should_save_model(roc_val_DLI, roc_val_DMI, roc_val_MLI, f1_val_DLI, f1_val_DMI, f1_val_MLI):
                torch.save(self.model, f"{self.model_save_folder}model_{self.args.network_type}.pt")
                print(f"Model saved at epoch {epoch}")

            if model_saver.should_stop():
                print(f"Early stopping triggered at epoch {epoch}")
                break

            print('The results of tttttttthis epoch: {:04d}'.format(epoch + 1),
                     '\n',
                    'train step => loss: {:.4f}'.format(total_loss.item()),
                    'auroc_DLI: {:.4f}'.format(roc_train_DLI),
                    'auroc_DMI: {:.4f}'.format(roc_train_DMI),
                    'auroc_MLI: {:.4f}'.format(roc_train_MLI),
                    '\n',
                    'loss_val_DLI: {:.4f}'.format(loss_val_DLI.item()),
                    'auroc_val_DLI: {:.4f}'.format(roc_val_DLI),
                    'auprc_val_DLI: {:.4f}'.format(prc_val_DLI),
                    'f1_val_DLI: {:.4f}'.format(f1_val_DLI),
                    '\n',
                    'loss_val_DMI: {:.4f}'.format(loss_val_DMI.item()),
                    'auroc_val_DMI: {:.4f}'.format(roc_val_DMI),
                    'auprc_val_DMI: {:.4f}'.format(prc_val_DMI),
                    'f1_val_DMI: {:.4f}'.format(f1_val_DMI),
                    '\n',
                    'loss_val_MLI: {:.4f}'.format(loss_val_MLI.item()),
                    'auroc_val_MLI: {:.4f}'.format(roc_val_MLI),
                    'auprc_val_MLI: {:.4f}'.format(prc_val_MLI),
                    'f1_val_MLI: {:.4f}'.format(f1_val_MLI),
                    '\n',
                    'time: {:.4f}s'.format(time.time() - t))

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Testing
        self.model = torch.load(f"{self.model_save_folder}model_{self.args.network_type}.pt")
### DLI____________________
        prediction, auroc_test_DLI, prc_test_DLI, f1_test_DLI, loss_test_DLI = self.score(self.test_loader_DLI, 'DLI')
        print('DLI: loss_test: {:.4f}'.format(loss_test_DLI.item()), 'auroc_test: {:.4f}'.format(auroc_test_DLI),
              'auprc_test: {:.4f}'.format(prc_test_DLI), 'f1_test: {:.4f}'.format(f1_test_DLI))
### DMI____________________
        prediction, auroc_test_DMI, prc_test_DMI, f1_test_DMI, loss_test_DMI = self.score(self.test_loader_DMI, 'DMI')
        print('DMI: loss_test: {:.4f}'.format(loss_test_DMI.item()), 'auroc_test: {:.4f}'.format(auroc_test_DMI),
              'auprc_test: {:.4f}'.format(prc_test_DMI), 'f1_test: {:.4f}'.format(f1_test_DMI))
### MLI____________________
        prediction, auroc_test_MLI, prc_test_MLI, f1_test_MLI, loss_test_MLI = self.score(self.test_loader_MLI, 'MLI')
        print('MLI: loss_test: {:.4f}'.format(loss_test_MLI.item()), 'auroc_test: {:.4f}'.format(auroc_test_MLI),
              'auprc_test: {:.4f}'.format(prc_test_MLI), 'f1_test: {:.4f}'.format(f1_test_MLI))

        # saving the results
        max_auroc = max(auroc_test_DLI, auroc_test_MLI, auroc_test_DMI)
        max_prc = max(prc_test_DLI, prc_test_MLI, prc_test_DMI)
        max_f1 = max(f1_test_DLI, f1_test_MLI, f1_test_DMI)
        results = {"auroc": max_auroc, "pr": max_prc, "f1": max_f1}

        if self.args.ratio:
            save_folder = f"results/{self.args.model}/network_{self.args.network_type}/{self.args.train_percent}/order_{len(self.args.layers_1)}/"
        else:
            save_folder = f"results/{self.args.model}/network_{self.args.network_type}/order_{len(self.args.layers_1)}/"

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        file_name = f"{save_folder}input_{self.args.input_type}_fold{self.args.fold_id}_lr{self.args.learning_rate}" \
                    f"_bs{self.args.batch_size}_hidden1_{self.args.hidden1}_hidden2_{self.args.hidden2}_dropout{self.args.dropout}.pt"
        torch.save(results, file_name)

        # saving embeddings
        with torch.no_grad():
            self.model.eval()
            latent_features = self.model.embed(self.propagation_matrix, self.features)
            embeddings = {"idxmap": self.idx_map, "emb": latent_features}

        if self.args.ratio:
            emb_folder = f"embeddings/{self.args.model}/network_{self.args.network_type}/{self.args.train_percent}/order_{len(self.args.layers_1)}/"
        else:
            emb_folder = f"embeddings/{self.args.model}/network_{self.args.network_type}/order_{len(self.args.layers_1)}/"

        if not os.path.exists(emb_folder):
            os.makedirs(emb_folder)

        file_name = f"{emb_folder}input_{self.args.input_type}_fold{self.args.fold_id}_lr{self.args.learning_rate}" \
                    f"_bs{self.args.batch_size}_hidden1_{self.args.hidden1}_hidden2_{self.args.hidden2}_dropout{self.args.dropout}.pt"
        torch.save(embeddings, file_name)

    def score(self, data_loader, task_type):
        """
        Scoring a neural network.
        :param indices: Indices of nodes involved in accuracy calculation.
        :return predictions: Probability for link existence
                roc_score: Area under ROC curve
                pr_score: Area under PR curve
                f1_score: F1 score
        """
        self.model.eval()
        y_pred = []
        y_label = []

        for i, (label, pairs) in enumerate(data_loader):
            label = label.to(self.device)
            if task_type == 'DLI':
                output, _, _, latent_feat = self.model(self.propagation_matrix, self.features, pairs)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(output.squeeze(), label.float().squeeze())
            elif task_type == 'DMI':
                _, output, _, latent_feat = self.model(self.propagation_matrix, self.features, pairs)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(output.squeeze(), label.float().squeeze())
            elif task_type == 'MLI':
                _, _, output, latent_feat = self.model(self.propagation_matrix, self.features, pairs)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(output.squeeze(), label.float().squeeze())

            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + output.flatten().tolist()

            outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])

        return y_pred, roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label,
                                                                                                          outputs), loss

