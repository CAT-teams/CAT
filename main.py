from __future__ import print_function

import argparse
import logging
import os
import random
import sys
import time
from inspect import getfile, currentframe
from os import getpid, environ
from os.path import dirname, abspath
from socket import gethostname
from sys import exit, argv

import matplotlib
import mlflow
import torch
import torch.backends.cudnn as cudnn
from torch import manual_seed as torch_manual_seed
from torch.cuda import is_available, set_device
from torch.cuda import manual_seed as cuda_manual_seed
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange

import Models
from customLoss import customLoss
from run import runTrain, runTest
from utils.dataset import loadModelNames, saveArgsToJSON, TqdmLoggingHandler, load_data, loadDatasets

matplotlib.use('agg')
import matplotlib.pyplot as plt


def parseArgs():
    modelNames = loadModelNames()
    datasets = loadDatasets()

    parser = argparse.ArgumentParser(description='Transform Coding')
    # general
    parser.add_argument('--data', type=str, required=True, help='location of the data corpus')
    parser.add_argument('--dataset', metavar='DATASET', default='cifar10', choices=datasets.keys(), help='dataset name')

    parser.add_argument('--gpu', type=str, default='0', help='gpu device id, e.g. 0,1,3')
    parser.add_argument('--seed', type=int, default=None, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--batch', default=128, type=int, help='batch size')
    parser.add_argument('--exp', type=str, default='', help='experiment name')
    parser.add_argument('--workers', default=30, type=int, help='Number of data loading workers (default: 2)')
    parser.add_argument('--print_freq', default=50, type=int, help='Number of batches between log messages')
    parser.add_argument('--save_freq', default=10, type=int, help='Number of batches between log messages')
    parser.add_argument('--pre-trained', default='preTrained', type=str, help='location of the pretrained models')
    parser.add_argument('--onlyInference', action='store_true', help='If use only inference')
    # optimization
    parser.add_argument('--lr', type=float, default=0.1, help='The learning rate.')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=4e-4, help='Weight decay (L2 penalty).')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma at scheduled epochs.')
    parser.add_argument('--schedule', type=int, nargs='+', default=[21, 35],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--regul', type=float, default=0, help='Regularization strength')
    parser.add_argument('--temp', type=float, default=-10, help='soft entropy temperature')
    parser.add_argument('--tempAnneal', type=float, default=1, help='soft entropy temperature annealing')
    parser.add_argument('--method', type=str, default='entropy',
                        choices=['compression', 'entropy'],
                        help='which method we use: [compression, entropy]')

    # algorithm
    parser.add_argument('--actBitwidth', default=32, type=float,
                        help='Quantization activation bitwidth (default: 32)')
    parser.add_argument('--weightBitwidth', default=32, type=float,
                        help='Quantization weight bitwidth (default: 32)')
    parser.add_argument('--model', '-a', metavar='MODEL', required=True, choices=modelNames,
                        help='model architecture: ' + ' | '.join(modelNames))
    parser.add_argument('--plotHist', action='store_true', help='Use entropy approximation')
    parser.add_argument('--clip', action='store_true', help='Use learnable clipping')
    parser.add_argument('--gradual', action='store_true', help='Use gradual quantization learnable')
    parser.add_argument('--noQuantEdges', action='store_true', help='Decide dont quant edges')
    parser.add_argument('--huffmanCode', action='store_true', help='If true show huffman entropy instead of theretical one')
    parser.add_argument('--stepSize', type=int, default=1, help='Steps per layer in gradual learning ')


    args = parser.parse_args()
    args.nClasses = datasets[args.dataset]

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        assert (args.model == 'resnet20' or args.model == 'resnet56')

    return args


if __name__ == '__main__':

    args = parseArgs()

    # set number of model output classes

    # run only in GPUs
    if not is_available():
        print('no gpu device available')
        exit(1)

    # update GPUs list
    if type(args.gpu) is not 'None':
        args.gpu = [int(i) for i in args.gpu.split(',')]

    args.device = 'cuda:' + str(args.gpu[0])

    # CUDA
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    random.seed(args.seed)
    set_device(args.gpu[0])
    cudnn.benchmark = True
    torch_manual_seed(args.seed)
    cudnn.enabled = True
    cuda_manual_seed(args.seed)

    # create folder
    baseFolder = dirname(abspath(getfile(currentframe())))
    args.time = time.strftime("%Y%m%d-%H%M%S")
    args.folderName = '{}_{}_{}_{}'.format(args.model, args.actBitwidth,
                                                    args.weightBitwidth,
                                                     args.time)
    args.save = '{}/results/{}'.format(baseFolder, args.folderName)
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # save args to JSON
    saveArgsToJSON(args)

    # Logger
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(filename=os.path.join(args.save, 'log.txt'), level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    th = TqdmLoggingHandler()
    th.setFormatter(logging.Formatter(log_format))
    log = logging.getLogger()
    log.addHandler(th)

    # Data
    testLoader, trainLoader, statsLoader = load_data(args, logging)

    # Model
    logging.info('{}\n'.format(' '.join(sys.argv)))
    logging.info('==> Building model..')
    modelClass = Models.__dict__[args.model]
    model = modelClass(args)

    if args.gradual:
        args.gradEpochs = args.stepSize * (model.depth)
    else:
        args.gradEpochs = 0

    assert args.epochs > args.gradEpochs  # ensure if gradual will pass in all layers

    # Load preTrained weights.
    logging.info('==> Resuming from checkpoint..')
    model.loadPreTrained(args.pre_trained, 'cpu')


    # criterion

    criterion = customLoss(args).cuda()

    # run statistic collection (only in 1 gpu)
    model = torch.nn.DataParallel(model, [args.gpu[0]])
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr, momentum=args.momentum, weight_decay=args.decay,
                                nesterov=True)

    logging.info('==>Start statistic')
    runTest(model, args, testLoader, 0, criterion, logging)
    logging.info('==>End statistic')


    for ops in model.module.modules():
        if isinstance(ops, torch.nn.ReLU):
            ops.collectStats = False
            # ACIQ initialization
            laplace = {1: 2.83, 2: 3.89, 3: 5.05, 4: 6.2, 5: 7.41, 6: 8.64, 7: 9.89, 8: 11.16}
            gaus = {1: 1.71, 2: 2.15, 3: 2.55, 4: 2.93, 5: 3.28, 6: 3.61, 7: 3.92, 8: 4.2}

            #            ops.c.data = ops.running_mean + (ops.running_b * laplace[args.actBitwidth])
            ops.c.data = ops.running_mean + (3 * ops.running_std)

            if not args.gradual:
                ops.quant = True

    if len(args.gpu) > 1:  # parallel
        model = torch.nn.DataParallel(model.module, args.gpu)
        model = model.cuda()

    gamma = (0.01 ** (1 / float(args.epochs - args.gradEpochs)))
    scheduler = ExponentialLR(optimizer, gamma=gamma)


    # log command line
    logging.info('CommandLine: {} PID: {} '
                 'Hostname: {} CUDA_VISIBLE_DEVICES {}'.format(argv, getpid(), gethostname(),
                                                               environ.get('CUDA_VISIBLE_DEVICES')))

    # mlflow
    mlflow.set_tracking_uri(os.path.join(baseFolder, 'mlruns_mxt'))

    mlflow.set_experiment(args.exp + '_' + args.model)

    if args.plotHist:
        global saveDic
        global num
        global step
        num = 0
        step = 0
        saveDic = args.save



    with mlflow.start_run(run_name="{}".format(args.folderName)):
        params = vars(args)
        for p in params:
            mlflow.log_param(p, params[p])
        start_epoch = 0
        layer = -1
        if not args.gradual:
            layer = model.module.depth - 1
        for epoch in trange(start_epoch, args.epochs):

            if (epoch < args.gradEpochs) and (epoch % args.stepSize == 0):
                layer += 1
                model.module.quantBlocks[layer].quant = True



            if not args.onlyInference:
                out = runTrain(model, args, trainLoader, epoch, optimizer, criterion, logging, layer)
                trainTotalLoss, trainCELoss, trainparamLoss, trainTop1, trainTop5 = out

            out = runTest(model, args, testLoader, epoch, criterion, logging)
            testTop1, testTop5, entropy, = out

            if args.plotHist and (epoch % args.save_freq == 0):

                def plotHistFunc(module, input, output):
                    global num
                    global step
                    global saveDic
                    fig = plt.figure()
                    plt.hist(input[0].view(-1).cpu(), bins=200)
                    dir = saveDic + '/plots'
                    if not os.path.exists(dir):
                        os.makedirs(dir)
                    fig.savefig(dir + '/' + str(num) + '_' + str(step) + '.png')
                    mlflow.log_artifact(dir + '/' + str(num) + '_' + str(step) + '.png', artifact_path=None)
                    num += 1


                # register hook
                handle = []
                for module in model.modules():
                    if isinstance(module, torch.nn.ReLU):
                        handle.append(module.register_forward_hook(plotHistFunc))
                model.eval()
                for batch_idx, (inputs, targets) in enumerate(statsLoader):
                    inputs, targets = inputs.cuda(), targets.cuda()
                    with torch.no_grad():
                        out, params = model(inputs)
                # unregister hook
                for h in handle:
                    h.remove()

                step += 1
                num = 0

            if mlflow.active_run() is not None:
                mlflow.log_metric('Test top1', testTop1)
                mlflow.log_metric('Test top5', testTop5)

                i = 0
                for m in model.module.modules():
                    if isinstance(m, torch.nn.ReLU):
                        mlflow.log_metric('Clip Layer ' + str(i), m.c.item())
                        i +=1


                for i in range(len(entropy)):
                    mlflow.log_metric('Entropy Layer ' + str(i), entropy[i].avg.item())


                mlflow.log_metric('Entropy Total ',
                                  (sum(d.sum for d in entropy) / sum(d.count for d in entropy)).item())

                if not args.onlyInference:
                    mlflow.log_metric('Train top1', trainTop1)
                    mlflow.log_metric('Train top5', trainTop5)
                    mlflow.log_metric('trainTotalLoss', trainTotalLoss)
                    mlflow.log_metric('trainCELoss', trainCELoss)
                    mlflow.log_metric('trainparamsLoss', trainparamLoss)
                # Save checkpoint.
                if epoch % args.save_freq == 0:
                    state = {
                        'net': model.state_dict(),
                        'acc': testTop1,
                        'epoch': epoch
                    }
                    torch.save(state, args.save + '/ckpt' + str(epoch) + '.t7')

            if epoch >= args.gradEpochs:
                scheduler.step()
        state = {
            'net': model.state_dict(),
            'acc': testTop1,
            'epoch': epoch
        }
        torch.save(state, args.save + '/ckpt' + str(epoch) + '.t7')
