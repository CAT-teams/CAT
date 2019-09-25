import time

import torch

from utils.meters import AverageMeter, accuracy


# For parallel implementation - transform dict to tensor
# 0 - act. 1- entropy .
def getParamsLoss(params, numDevices):
    elems = 0
    entr = 0
    for p in params:
        for i in range(numDevices):
            elems += p[i * 2]
            entr += p[i * 2] * p[(i * 2) + 1]

    return entr / elems # sum(d[1]*d[0] for d in params) / sum(d[0] for d in params)



def runTrain(model, args, trainLoader, epoch, optimizer, criterion, logging, layer):
    model.train()
    batch_time = AverageMeter()
    totalLosses = AverageMeter()
    ceLosses = AverageMeter()
    paramsLosses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainLoader):


        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        out,params = model(inputs)



        totalLoss, crossEntropyLoss, paramsLoss = criterion(out, targets, getParamsLoss(params[:layer + 1], len(model.device_ids)) )

        totalLoss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(out, targets, topk=(1, 5))
        totalLosses.update(totalLoss.item(), inputs.size(0))
        ceLosses.update(crossEntropyLoss.item(), inputs.size(0))
        paramsLosses.update(paramsLoss.item(), inputs.size(0))
        top1.update(float(prec1), inputs.size(0))
        top5.update(float(prec5), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            logging.info('Epoch Train: [{}]\t'
                         'Train: [{}/{}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Total Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Cross Entropy Loss {CEloss.val:.4f} ({CEloss.avg:.4f})\t'
                         'paramsLoss Loss {paramsLoss.val:.4f} ({paramsLoss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, batch_idx + 1, len(trainLoader), batch_time=batch_time, loss=totalLosses,
                CEloss=ceLosses, paramsLoss=paramsLosses, top1=top1, top5=top5))

    return totalLosses.avg, ceLosses.avg, paramsLosses.avg, top1.avg, top5.avg


def runTest(model, args, testLoader, epoch, criterion, logging):
    model.eval()
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    entropy = [AverageMeter() for i in range(model.module.depth)]

    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(testLoader):
        inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            out,params = model(inputs)

            if len(model.device_ids) > 1:  # parallel
                assert len(params[0]) % 2 == 0
                for p in params:
                    p[0:2] = sum(list(torch.split(p, 2))) / len(args.gpu)
            # For parallel implementation - transform dict to tensor
            # 0 - maxStdRatio. 1- MaxMeanRatio . 2- kurtosis. 3 -entropy. 4-act. 5-quantError
        for i in range(model.module.depth):
            entropy[i].update(params[i][1],params[i][0])

        # measure accuracy and record loss
        prec1, prec5 = accuracy(out, targets, topk=(1, 5))
        top1.update(float(prec1), inputs.size(0))
        top5.update(float(prec5), inputs.size(0))

    # measure elapsed time
    batch_time.update(time.time() - end)
    logging.info('Epoch Test: [{}]\t'
                 'Time ({batch_time.avg:.3f})\t'
                 'Entropy {ent} \t'
                 # 'Kurtosis {kurt} \t'
                 # 'maxStdRatio {mxstd} \t'
                 # 'maxMeanRatio {mxmean} \t'
                 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                 'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        epoch, batch_time=batch_time,
        ent=sum(d.sum for d in entropy)/ sum(d.count for d in entropy),
         top1=top1, top5=top5))

    return top1.avg, top5.avg,entropy


