import argparse
import os
import time
from collections import OrderedDict
from copy import deepcopy

import medmnist
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from acsconv.converters import ACSConverter, Conv2_5dConverter, Conv3dConverter
from medmnist import INFO, Evaluator
from models.models import ResNet18, ResNet50
from bayesian_torch.models.dnn_to_bnn import get_kl_loss
from tensorboardX import SummaryWriter
from tqdm import trange
from utils.utils import Transform3D, model_to_syncbn
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn


def main(data_flag, output_root, num_epochs, gpu_ids, batch_size, conv, pretrained_3d, download, model_flag, as_rgb, shape_transform, model_path, run):

    lr = 0.001
    gamma=0.1
    milestones = [0.5 * num_epochs, 0.75 * num_epochs]

    info = INFO[data_flag]
    task = info['task']
    n_channels = 3 if as_rgb else info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])
    
    str_ids = gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_ids[0])

    device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu') 
    
        
    output_root = os.path.join(output_root, data_flag, time.strftime("%y%m%d_%H%M%S"))
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    print('==> Preparing data...')

    train_transform = Transform3D(mul='random') if shape_transform else Transform3D()
    eval_transform = Transform3D(mul='0.5') if shape_transform else Transform3D()
     
    train_dataset = DataClass(
        split='train', transform=train_transform, download=download, as_rgb=as_rgb
    )
    train_dataset_at_eval = DataClass(
        split='train', transform=eval_transform, download=download, as_rgb=as_rgb
    )
    val_dataset = DataClass(
        split='val', transform=eval_transform, download=download, as_rgb=as_rgb
    )
    test_dataset = DataClass(
        split='test', transform=eval_transform, download=download, as_rgb=as_rgb
    )

    
    train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset_at_eval,
                                batch_size=batch_size,
                                shuffle=False)
    val_loader = data.DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False)

    print('==> Building and training model...')

    if model_flag == 'sviresnet18':
        model = ResNet18(in_channels=n_channels, num_classes=n_classes)
    elif model_flag == 'sviresnet50':
        model =  ResNet50(in_channels=n_channels, num_classes=n_classes)
    else:
        raise NotImplementedError

    if conv=='ACSConv':
        model = model_to_syncbn(ACSConverter(model))
    if conv=='Conv2_5d':
        model = model_to_syncbn(Conv2_5dConverter(model))
    if conv=='Conv3d':
        if pretrained_3d == 'i3d':
            model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=-3))
        else:
            model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=None))
    
    const_bnn_prior_parameters = {
            "prior_mu": 0.0,
            "prior_sigma": 1.0,
            "posterior_mu_init": 0.0,
            "posterior_rho_init": -3.0,
            "type": "Flipout",  # Flipout or Reparameterization
            "moped_enable": True,  # True to initialize mu/sigma from the pretrained dnn weights
            "moped_delta": 0.5,
        }

    dnn_to_bnn(model, const_bnn_prior_parameters)
    model = model.to(device)

    train_evaluator = medmnist.Evaluator(data_flag, 'train')
    val_evaluator = medmnist.Evaluator(data_flag, 'val')
    test_evaluator = medmnist.Evaluator(data_flag, 'test')

    kl_weight = 0.0
    svi_train_samples = 1
    svi_val_samples = 1
    svi_test_samples = 20

    criterion = nn.CrossEntropyLoss()

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device)['net'], strict=True)
        train_metrics = test(
            model, train_evaluator, train_loader_at_eval, 
            criterion, device, run, svi_train_samples, kl_weight, 
            output_root
        )
        val_metrics = test(
            model, val_evaluator, val_loader, 
            criterion, device, run, svi_val_samples, kl_weight, 
            output_root
        )
        test_metrics = test(
            model, test_evaluator, test_loader, criterion, 
            device, run, svi_test_samples, kl_weight, 
            output_root
        )

        print('train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2]) + \
              'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2]) + \
              'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2]))

    if num_epochs == 0:
        return


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    logs = ['loss', 'auc', 'acc']
    train_logs = ['train_'+log for log in logs]
    val_logs = ['val_'+log for log in logs]
    test_logs = ['test_'+log for log in logs]
    log_dict = OrderedDict.fromkeys(train_logs+val_logs+test_logs, 0)
    
    writer = SummaryWriter(log_dir=os.path.join(output_root, 'Tensorboard_Results'))

    best_auc = 0
    best_epoch = 0
    best_model = deepcopy(model)

    global iteration
    iteration = 0

    for epoch in trange(num_epochs):
        
        train_loss = train(
            model, train_loader, criterion, optimizer, 
            device, writer, kl_weight, svi_train_samples
        )
        train_metrics = test(
            model, train_evaluator, train_loader_at_eval, 
            criterion, device, run, svi_train_samples, kl_weight
        )
        val_metrics = test(
            model, val_evaluator, val_loader, 
            criterion, device, run, svi_val_samples, kl_weight
        )
        test_metrics = test(
            model, test_evaluator, test_loader, criterion, 
            device, run, svi_test_samples, kl_weight
        )
        
        scheduler.step()
        
        for i, key in enumerate(train_logs):
            log_dict[key] = train_metrics[i]
        for i, key in enumerate(val_logs):
            log_dict[key] = val_metrics[i]
        for i, key in enumerate(test_logs):
            log_dict[key] = test_metrics[i]

        for key, value in log_dict.items():
            writer.add_scalar(key, value, epoch)
            
        cur_auc = val_metrics[1]
        if cur_auc > best_auc:
            best_epoch = epoch
            best_auc = cur_auc
            best_model = deepcopy(model)

            print('cur_best_auc:', best_auc)
            print('cur_best_epoch', best_epoch)

    state = {
        'net': model.state_dict(),
    }

    path = os.path.join(output_root, 'best_model.pth')
    torch.save(state, path)

    train_metrics = test(
        best_model, train_evaluator, train_loader_at_eval, 
        criterion, device, run, svi_train_samples, kl_weight, 
        output_root
    )
    val_metrics = test(
        best_model, val_evaluator, val_loader, criterion, 
        device, run, svi_val_samples, kl_weight, output_root
    )
    test_metrics = test(
        best_model, test_evaluator, test_loader, criterion, 
        device, run, svi_test_samples, kl_weight, output_root
    )

    train_log = 'train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2])
    val_log = 'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2])
    test_log = 'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2])

    log = '%s\n' % (data_flag) + train_log + val_log + test_log + '\n'
    print(log)
    
    with open(os.path.join(output_root, '%s_log.txt' % (data_flag)), 'a') as f:
        f.write(log)        
            
    writer.close()


def train(model, train_loader, criterion, optimizer, device, writer, kl_weight, num_iters):
    total_loss = []
    global iteration

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs = inputs.to(device)

        targets = torch.squeeze(targets, 1).long().to(device)

        list_out = [model(inputs) for _ in range(num_iters)]
        loss = 0.0

        for out in list_out:
            loss += criterion(out, targets) / num_iters

        if kl_weight != 0.0:
            kl = get_kl_loss(model)
            loss = loss + kl / inputs.size(0)

        total_loss.append(loss.item())
        writer.add_scalar('train_loss_logs', loss.item(), iteration)
        iteration += 1

        loss.backward()
        optimizer.step()
    
    epoch_loss = sum(total_loss)/len(total_loss)
    return epoch_loss


def test(model, evaluator, data_loader, criterion, device, run, svi_samples, kl_weight, save_folder=None):

    model.eval()

    total_loss = []
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)

            svi_out = torch.vstack([model(inputs).unsqueeze(0) for _ in range(svi_samples)])
            targets = torch.squeeze(targets, 1).long().to(device)
            svi_act = F.softmax(svi_out, dim=2)

            var_act, mean_act = torch.var_mean(svi_act, dim=0)
            conf, preds = torch.max(mean_act, dim=1)

            loss = criterion(svi_out.mean(dim=0), targets)
            kl = sum(m.kl_divergence() for m in model.modules() if hasattr(m, "kl_divergence"))
            loss += kl_weight * kl

            total_loss.append(loss.item())

            y_score = torch.cat((y_score, svi_act.mean(dim=0)), 0)

        y_score = y_score.detach().cpu().numpy()
        auc, acc = evaluator.evaluate(y_score, save_folder, run)

        test_loss = sum(total_loss) / len(total_loss)

        return [test_loss, auc, acc]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Baseline model of MedMNIST3D')

    parser.add_argument('--data_flag',
                        default='organmnist3d',
                        type=str)
    parser.add_argument('--output_root',
                        default='./output',
                        help='output root, where to save models',
                        type=str)
    parser.add_argument('--num_epochs',
                        default=100,
                        help='num of epochs of training, the script would only test model if set num_epochs to 0',
                        type=int)
    parser.add_argument('--gpu_ids',
                        default='0',
                        type=str)
    parser.add_argument('--batch_size',
                        default=32,
                        type=int)
    parser.add_argument('--conv',
                        default='ACSConv',
                        help='choose converter from Conv2_5d, Conv3d, ACSConv',
                        type=str)
    parser.add_argument('--pretrained_3d',
                        default='i3d',
                        type=str)
    parser.add_argument('--download',
                        action="store_true")
    parser.add_argument('--as_rgb',
                        help='to copy channels, tranform shape 1x28x28x28 to 3x28x28x28',
                        action="store_true")
    parser.add_argument('--shape_transform',
                        help='for shape dataset, whether multiply 0.5 at eval',
                        action="store_true")
    parser.add_argument('--model_path',
                        default=None,
                        help='root of the pretrained model to test',
                        type=str)
    parser.add_argument('--model_flag',
                        default='resnet18',
                        help='choose backbone, resnet18/resnet50',
                        type=str)
    parser.add_argument('--run',
                        default='model1',
                        help='to name a standard evaluation csv file, named as {flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv',
                        type=str)


    args = parser.parse_args()
    data_flag = args.data_flag
    output_root = args.output_root
    num_epochs = args.num_epochs
    gpu_ids = args.gpu_ids
    batch_size = args.batch_size
    conv = args.conv
    pretrained_3d = args.pretrained_3d
    download = args.download
    model_flag = args.model_flag
    as_rgb = args.as_rgb
    model_path = args.model_path
    shape_transform = args.shape_transform
    run = args.run

    main(data_flag, output_root, num_epochs, gpu_ids, batch_size, conv, pretrained_3d, download, model_flag, as_rgb, shape_transform, model_path, run)