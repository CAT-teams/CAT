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

def soft_entropy( x,  bits=8, temp=-10):
    if torch.numel(torch.unique(x)) == 1:
        return 0
    bins = int(2 ** bits)
    centers = torch.linspace(0, bins - 1, bins).cuda()
    x = (x.repeat(bins, 1).t() - centers) ** 2

    x = temp * x
    x = torch.nn.functional.softmax(x, 1, _stacklevel=5)
    x = torch.sum(x, dim=0) / x.shape[0]
    x[x == 0] = 1  # hack
    x = -x * torch.log(x)
    return torch.sum(x) / np.log(2)

def shannon_entropy2(x, base=2, bits=8):
    #   pk = torch.bincount(torch.round(x).long().flatten())

    pk = torch.histc(x, max=base ** bits - 1, min=0, bins=base ** bits)

    pk = pk.float() / torch.sum(pk).float()

    pk[pk == 0] = 1  # HACK
    vec = -pk * torch.log(pk)
    return torch.sum(vec) / np.log(base)
	
def runTrain(model, args, trainLoader, epoch, optimizer, criterion, logging, layer):
    model.train()
    batch_time = AverageMeter()
    totalLosses = AverageMeter()
    ceLosses = AverageMeter()
    paramsLosses = AverageMeter()
		paramsLosses2 = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainLoader):


        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        out,params = model(inputs)

		if args.regul2 >0:
			totalElems = 0
			EntrTotal = []
			for m in model.modules():
				if isinstance(m, torch.nn.Conv2d):
					elems = torch.numel(m.weight)
					scale = (torch.max(m.weight) - torch.min(m.weight)) / ((2. ** 8) - 1.)
					qweight = (m.weight.view(-1) - torch.min(m.weight)) / scale
					numIdxs = int(elems)
					idx = torch.randperm(numIdxs, device=m.weight.device)[:int(numIdxs / 20)]
					qweight = qweight[idx]
					EntrTotal.append(soft_entropy(qweight,bits=8,temp=-10) * elems)
					totalElems += elems
			EntrTotal = sum(EntrTotal) / totalElems
		else:
			EntrTotal = 0
			
        totalLoss, crossEntropyLoss, paramsLoss, paramLoss2 = criterion(out, targets, getParamsLoss(params[:layer + 1], len(model.device_ids)) ,EntrTotal)

      #  totalLoss, crossEntropyLoss, paramsLoss = criterion(out, targets, getParamsLoss(params[:layer + 1], len(model.device_ids)) )

        totalLoss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(out, targets, topk=(1, 5))
        totalLosses.update(totalLoss.item(), inputs.size(0))
        ceLosses.update(crossEntropyLoss.item(), inputs.size(0))
        paramsLosses.update(paramsLoss.item(), inputs.size(0))
		paramsLosses2.update(paramLoss2.item(), inputs.size(0))
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
                         'paramsLoss Loss2 {paramsLoss2.val:.4f} ({paramsLoss2.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, batch_idx + 1, len(trainLoader), batch_time=batch_time, loss=totalLosses,
                CEloss=ceLosses, paramsLoss=paramsLosses,paramsLoss2 = paramsLosses2, top1=top1, top5=top5))

    return totalLosses.avg, ceLosses.avg, paramsLosses.avg, paramsLosses2.avg, top1.avg, top5.avg


def runTest(model, args, testLoader, epoch, criterion, logging):
    model.eval()
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    entropy = [AverageMeter() for i in range(model.module.depth)]
    entropyW = [AverageMeter() for i in range(20)] #change to parameter - this is only for resnet18
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

		if args.regul2 >0:
			for i,m in enumerate(model.modules()):
				if isinstance(m, torch.nn.Conv2d):
					elems = torch.numel(m.weight)
					scale = (torch.max(m.weight) - torch.min(m.weight)) / ((2. ** 8) - 1.)
					qweight = (m.weight.view(-1) - torch.min(m.weight)) / scale
					numIdxs = int(elems)
					idx = torch.randperm(numIdxs, device=m.weight.device)[:int(numIdxs / 20)]
					qweight = qweight[idx]			
					entropyW[i].update(shannon_entropy2(qweight,bits=8),elems)
			
        # measure accuracy and record loss
        prec1, prec5 = accuracy(out, targets, topk=(1, 5))
        top1.update(float(prec1), inputs.size(0))
        top5.update(float(prec5), inputs.size(0))

    # measure elapsed time
    batch_time.update(time.time() - end)
    logging.info('Epoch Test: [{}]\t'
                 'Time ({batch_time.avg:.3f})\t'
                 'Entropy {ent} \t'
				 'EntropyW {entW} \t'
                 # 'Kurtosis {kurt} \t'
                 # 'maxStdRatio {mxstd} \t'
                 # 'maxMeanRatio {mxmean} \t'
                 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                 'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        epoch, batch_time=batch_time,
        ent=sum(d.sum for d in entropy)/ sum(d.count for d in entropy),
		entW=sum(d.sum for d in entropyW)/ sum(d.count for d in entropyW),
         top1=top1, top5=top5))

    return top1.avg, top5.avg,entropy


