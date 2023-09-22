import argparse
import torch
import torch.nn as nn
import numpy as np
import json
import warnings
from sklearn.utils import class_weight
from utils import setup_seed, print_metrics_binary, imp_con_loss, pre_con_loss, mul_con_loss, mre_f, device
from model import Model
from data_extraction import data_process_mimic3


def train(train_loader,
          valid_loader,
          demographic_data,
          diagnosis_data,
          procedure_data,
          idx_list,
          x_dim,
          base_dim,
          d_model_att,
          n_views,
          seq_len,
          d_model_tf,
          n_heads,
          n_layers,
          ff_dim,
          dropout,
          lambda_ce,
          lambda_mae,
          lambda_pre,
          lambda_imp,
          lr,
          seed,
          epochs,
          file_name,
          device):

    model = Model(x_dim, base_dim, d_model_att, n_views, seq_len,
                  d_model_tf, n_heads, n_layers, ff_dim, dropout).to(device)
    opt_model = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(opt_model, milestones=[40, 60, 80, 90], gamma=0.5)

    setup_seed(seed)
    train_loss = []
    valid_loss_mae = []
    valid_loss_mre = []
    best_epoch = 0
    max_auroc = 0

    for each_epoch in range(epochs):
        batch_loss = []
        model.train()

        for step, (batch_x, batch_y, sorted_length, batch_ts, batch_name) in enumerate(train_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.to(device)

            mask = torch.where(batch_x != -1, torch.ones(batch_x.shape).to(device),
                               torch.zeros(batch_x.shape).to(device))

            batch_demo = []
            batch_diag = []
            batch_pro = []
            for i in range(len(batch_name)):
                cur_id, cur_ep, _ = batch_name[i].split('_', 2)
                cur_idx = cur_id + '_' + cur_ep
                idx = idx_list.index(cur_idx) if cur_idx in idx_list else None

                cur_demo = torch.tensor(demographic_data[idx], dtype=torch.float32)
                cur_diag = torch.tensor(diagnosis_data[idx], dtype=torch.float32)
                cur_pro = torch.tensor(procedure_data[idx], dtype=torch.float32)

                batch_demo.append(cur_demo)
                batch_diag.append(cur_diag)
                batch_pro.append(cur_pro)

            batch_demo = torch.stack(batch_demo).to(device)
            batch_diag = torch.stack(batch_diag).to(device)
            batch_pro = torch.stack(batch_pro).to(device)

            batch_base = torch.cat((batch_demo, batch_diag, batch_pro), 1)
            output, x_final, gnn_views, attns = model(batch_x, batch_base, sorted_length)

            batch_y = batch_y.long()
            y_out = batch_y.cpu().numpy()
            class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_out),
                                                              y=y_out)
            class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
            ce_f = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
            mae_f = torch.nn.L1Loss(reduction='mean')

            loss_ce = ce_f(output, batch_y)
            loss_mae = mae_f(mask * x_final, mask * batch_x)

            loss_precl = mul_con_loss(pre_con_loss, gnn_views, y_label=batch_y)
            loss_impcl = mul_con_loss(imp_con_loss, gnn_views, adj=attns)

            loss_pre = lambda_ce * loss_ce + (1 - lambda_ce) * loss_precl
            loss_imp = lambda_mae * loss_mae + (1 - lambda_mae) * loss_impcl

            loss = lambda_pre * loss_pre + lambda_imp * loss_imp
            batch_loss.append(loss.cpu().detach().numpy())

            opt_model.zero_grad()
            loss.backward()
            opt_model.step()

        train_loss.append(np.mean(np.array(batch_loss)))

        # scheduler.step()
        with torch.no_grad():
            y_true = []
            y_pred = []
            batch_loss_mae = []
            batch_loss_mre = []
            model.eval()

            for step, (batch_x, batch_y, sorted_length, batch_ts, batch_name) in enumerate(valid_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.to(device)

                mask = torch.where(batch_x != -1, torch.ones(batch_x.shape).to(device),
                                   torch.zeros(batch_x.shape).to(device))

                batch_demo = []
                batch_diag = []
                batch_pro = []
                for i in range(len(batch_name)):
                    cur_id, cur_ep, _ = batch_name[i].split('_', 2)
                    cur_idx = cur_id + '_' + cur_ep
                    idx = idx_list.index(cur_idx) if cur_idx in idx_list else None

                    cur_demo = torch.tensor(demographic_data[idx], dtype=torch.float32)
                    cur_diag = torch.tensor(diagnosis_data[idx], dtype=torch.float32)
                    cur_pro = torch.tensor(procedure_data[idx], dtype=torch.float32)

                    batch_demo.append(cur_demo)
                    batch_diag.append(cur_diag)
                    batch_pro.append(cur_pro)

                batch_demo = torch.stack(batch_demo).to(device)
                batch_diag = torch.stack(batch_diag).to(device)
                batch_pro = torch.stack(batch_pro).to(device)

                batch_base = torch.cat((batch_demo, batch_diag, batch_pro), 1)
                output, x_final, _, _ = model(batch_x, batch_base, sorted_length)

                loss_mae = mae_f(mask * x_final, mask * batch_x)
                loss_mre = mre_f(mask * x_final, mask * batch_x)
                batch_y = batch_y.long()

                batch_loss_mae.append(loss_mae.cpu().detach().numpy())
                batch_loss_mre.append(loss_mre.cpu().detach().numpy())

                y_pred.append(output)
                y_true.append(batch_y)

            y_pred = torch.cat(y_pred, 0)
            y_true = torch.cat(y_true, 0)
            test_y_pred = y_pred.cpu().detach().numpy()
            test_y_true = y_true.cpu().detach().numpy()
            ret = print_metrics_binary(test_y_true, test_y_pred)

            valid_loss_mae.append(np.mean(np.array(batch_loss_mae)))
            valid_loss_mre.append(np.mean(np.array(batch_loss_mre)))

            cur_auroc = ret['auroc']
            cur_auprc = ret['auprc']

            if cur_auroc > max_auroc:
                max_auroc = cur_auroc
                best_epoch = each_epoch
                state = {
                    'net': model.state_dict(),
                    'optimizer': opt_model.state_dict(),
                    'epoch': each_epoch
                }
                torch.save(state, file_name)

    return best_epoch


def test(test_loader,
         demographic_data,
         diagnosis_data,
         procedure_data,
         idx_list,
         x_dim,
         base_dim,
         d_model_att,
         n_views,
         seq_len,
         d_model_tf,
         n_heads,
         n_layers,
         ff_dim,
         dropout,
         seed,
         file_name,
         device):

    setup_seed(seed)
    model = Model(x_dim, base_dim, d_model_att, n_views, seq_len,
                  d_model_tf, n_heads, n_layers, ff_dim, dropout).to(device)
    checkpoint = torch.load(file_name)
    model.load_state_dict(checkpoint['net'])
    model.eval()

    batch_loss_mae = []
    batch_loss_mre = []
    test_loss_mae = []
    test_loss_mre = []
    y_true = []
    y_pred = []

    for step, (batch_x, batch_y, sorted_length, batch_ts, batch_name) in enumerate(test_loader):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.to(device)

        mask = torch.where(batch_x != -1, torch.ones(batch_x.shape).to(device),
                           torch.zeros(batch_x.shape).to(device))

        batch_demo = []
        batch_diag = []
        batch_pro = []
        for i in range(len(batch_name)):
            cur_id, cur_ep, _ = batch_name[i].split('_', 2)
            cur_idx = cur_id + '_' + cur_ep
            idx = idx_list.index(cur_idx) if cur_idx in idx_list else None

            cur_demo = torch.tensor(demographic_data[idx], dtype=torch.float32)
            cur_diag = torch.tensor(diagnosis_data[idx], dtype=torch.float32)
            cur_pro = torch.tensor(procedure_data[idx], dtype=torch.float32)

            batch_demo.append(cur_demo)
            batch_diag.append(cur_diag)
            batch_pro.append(cur_pro)

        batch_demo = torch.stack(batch_demo).to(device)
        batch_diag = torch.stack(batch_diag).to(device)
        batch_pro = torch.stack(batch_pro).to(device)

        batch_base = torch.cat((batch_demo, batch_diag, batch_pro), 1)
        output, x_final, _, _ = model(batch_x, batch_base, sorted_length)

        mae_f = torch.nn.L1Loss(reduction='mean')
        loss_mae = mae_f(mask * x_final, mask * batch_x)
        loss_mre = mre_f(mask * x_final, mask * batch_x)
        batch_y = batch_y.long()

        batch_loss_mae.append(loss_mae.cpu().detach().numpy())
        batch_loss_mre.append(loss_mre.cpu().detach().numpy())

        y_pred.append(output)
        y_true.append(batch_y)

    y_pred = torch.cat(y_pred, 0)
    y_true = torch.cat(y_true, 0)
    test_y_pred = y_pred.cpu().detach().numpy()
    test_y_true = y_true.cpu().detach().numpy()
    ret = print_metrics_binary(test_y_true, test_y_pred)
    cur_auroc = ret['auroc']
    cur_auprc = ret['auprc']

    test_loss_mae.append(np.mean(np.array(batch_loss_mae)))
    test_loss_mre.append(np.mean(np.array(batch_loss_mre)))
    cur_mae = test_loss_mae[-1]
    cur_mre = test_loss_mre[-1]

    results = {'auroc': cur_auroc, 'auprc': cur_auprc, 'mae': cur_mae, 'mre': cur_mre}

    return results


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # Define Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_dim", type=int, default=17)
    parser.add_argument("--d_model_att", type=int, default=17)
    parser.add_argument("--n_views", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=200)
    parser.add_argument("--d_model_tf", type=int, default=24)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--ff_dim", type=int, default=17)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lambda_ce", type=float, default=0.8)
    parser.add_argument("--lambda_mae", type=float, default=0.8)
    parser.add_argument("--lambda_pre", type=float, default=0.7)
    parser.add_argument("--lambda_imp", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--base_dim", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--time_length", type=int)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_s_path1", type=str)
    parser.add_argument("--data_s_path2", type=str)
    parser.add_argument("--data_s_path3", type=str)
    parser.add_argument("--data_idx_path", type=str)
    parser.add_argument("--file_name", type=str)
    args = parser.parse_args()

    x_dim = args.x_dim
    d_model_att = args.d_model_att
    n_views = args.n_views
    seq_len = args.seq_len
    d_model_tf = args.d_model_tf
    n_heads = args.n_heads
    n_layers = args.n_layers
    ff_dim = args.ff_dim
    dropout = args.dropout
    lambda_ce = args.lambda_ce
    lambda_mae = args.lambda_mae
    lambda_pre = args.lambda_pre
    lambda_imp = args.lambda_imp
    lr = args.lr
    base_dim = args.base_dim
    seed = args.seed
    epochs = args.epochs
    time_length = args.time_length
    data_path = args.data_path
    data_s_path1 = args.data_s_path1
    data_s_path2 = args.data_s_path2
    data_s_path3 = args.data_s_path3
    data_idx_path = args.data_idx_path
    file_name = args.file_name

    train_loader, valid_loader, test_loader = data_process_mimic3(data_path, time_length)
    with open(data_s_path1, 'r') as f:
        demographic_data = json.load(f)
    with open(data_s_path2, 'r') as f:
        diagnosis_data = json.load(f)
    with open(data_s_path3, 'r') as f:
        procedure_data = json.load(f)
    with open(data_idx_path, 'r') as f:
        idx_list = json.load(f)

    best_epoch = train(train_loader, valid_loader, demographic_data, diagnosis_data, procedure_data, idx_list, x_dim, base_dim, d_model_att, n_views, seq_len,
                  d_model_tf, n_heads, n_layers, ff_dim, dropout, lambda_ce, lambda_mae, lambda_pre, lambda_imp, lr, seed, epochs, file_name, device)
    results = test(test_loader, demographic_data, diagnosis_data, procedure_data, idx_list, x_dim, base_dim, d_model_att, n_views, seq_len,
                  d_model_tf, n_heads, n_layers, ff_dim, dropout, seed, file_name, device)
    print(results)

