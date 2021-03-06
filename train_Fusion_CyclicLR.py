import os
# os.environ['CUDA_VISIBLE_DEVICES'] =  '6'
import sys
sys.path.append("..")
import argparse
from process.data_fusion import *
from process.augmentation import *
from metric import *
from loss import *

from utils import convert


def get_model(model_name, num_class):
    if model_name == 'baseline':
        from model_fusion.model_baseline_SEFusion import FusionNet
    elif model_name == 'model_A':
        from model_fusion.FaceBagNet_model_A_SEFusion import FusionNet
    elif model_name == 'model_B':
        from model_fusion.FaceBagNet_model_B_SEFusion import FusionNet
    elif model_name == 'FeatherNet':
        from model_fusion.FaceBagNet_model_FTB_SEFusion import FusionNet
    elif model_name == 'repvgg':
        from model_fusion.FacebagNet_model_RepVGG_SEFusion import FusionNet
    net = FusionNet(num_class=num_class)
    return net

def run_train(config):
    out_dir = config.saved_path
    config.model_name = config.model + '_' + config.image_mode + '_' + str(config.image_size)
    out_dir = os.path.join(out_dir,config.model_name)
    initial_checkpoint = config.pretrained_model

    ## setup  -----------------------------------------------------------------------------
    if not os.path.exists(out_dir +'/checkpoint'):
        os.makedirs(out_dir +'/checkpoint')
    if not os.path.exists(out_dir +'/backup'):
        os.makedirs(out_dir +'/backup')
    if not os.path.exists(out_dir +'/backup'):
        os.makedirs(out_dir +'/backup')

    log = Logger()
    log.open(os.path.join(out_dir,config.model_name+'.txt'),mode='a')
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')
    log.write('\t<additional comments>\n')
    log.write('\t  ... xxx baseline  ... \n')
    log.write('\n')

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    train_dataset = FDDataset(mode = 'train', modality=config.image_mode,image_size=config.image_size,
                              fold_index=config.train_fold_index)
    train_loader  = DataLoader(train_dataset,
                                shuffle=True,
                                batch_size  = config.batch_size,
                                drop_last   = True,
                                num_workers = 8)

    valid_dataset = FDDataset(mode = 'val', modality=config.image_mode,image_size=config.image_size,
                              fold_index=config.train_fold_index)
    valid_loader  = DataLoader( valid_dataset,
                                shuffle=False,
                                batch_size  = config.batch_size // 4,
                                drop_last   = False,
                                num_workers = 8)
    #### Distribution Data #######
    sum1 = 0
    dict_samples = train_dataset.analyze()
    nclass = len(dict_samples.keys())
    log.write('Data distribution for valid: %s' %dict_samples)
    # log.write()
    sum1 = sum(dict_samples.values())
    max1 = max(dict_samples.values())
    # print('\n')
    # exit()
    weights_loss = [ sum1/i for i in dict_samples.values() ]
    weights_loss = torch.FloatTensor(np.array(weights_loss))

    if config.criterion == 'ce':
        criterion = {
            'ce': softmax_cross_entropy_criterion,
        }
    elif config.criterion == 'triplet':
        criterion = {
            'triplet': TripletLoss('cuda' if torch.cuda.is_available() else 'cpu'),
            'ce': softmax_cross_entropy_criterion,
        }
    elif config.criterion == 'crl':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        criterion_class = FocalLoss(2, 0.5).to(device)
        # CRL loss
        nuy = 0.1
        lalpha = (1-(np.array(weights_loss)/max1).mean())*nuy
        criterion_metric = CRClassLoss(nclass=nclass, 
                            comparison='relative', 
                            p=1/nclass, k=None, margin=0.5, 
                            device=device)
        criterion = {
            'ce': softmax_cross_entropy_criterion,
            'crl': criterion_metric,
            'focal': criterion_class,
        }

    assert(len(train_dataset)>=config.batch_size)
    log.write('\nbatch_size = %d\n'%(config.batch_size))
    log.write('train_dataset : \n%d\n'%(len(train_dataset)))
    log.write('valid_dataset : \n%d\n'%(len(valid_dataset)))
    log.write('\n')
    log.write('** net setting **\n')

    net = get_model(model_name=config.model, num_class=2)
    log.write('Model Architecture: \n',  net)
    # net = torch.nn.DataParallel(net)
    net =  net.cuda()
    
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print('total_params: ',pytorch_total_params)
    # torch.save(net.state_dict(), 'RepVGG-training.pth')
    # exit(0)

    if initial_checkpoint is not None:
        # initial_checkpoint = os.path.join(out_dir +'/checkpoint',initial_checkpoint)
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage), strict=False)

    log.write('%s\n'%(type(net)))
    log.write('criterion=%s\n'%config.criterion)
    log.write('\n')

    iter_smooth = 20
    start_iter = 0
    log.write('\n')

    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write('                               |--------------------------- VALID -----------------------------|--------- TRAIN/BATCH -------| \n')
    log.write('model_name   lr   iter  epoch  |  loss   acer   acc  tpr@fpr:1e-2  tpr@fpr:1e-3  tpr@fpr:1e-4  | loss_{}  loss_triplet  acc  |  time   \n'.format(config.criterion))
    log.write('-----------------------------------------------------------------------------------------------------------------------\n')

    train_loss   = np.zeros(6,np.float32)
    valid_loss   = np.zeros(6,np.float32)
    batch_loss   = np.zeros(6,np.float32)
    iter = 0
    i    = 0

    start = timer()
    #-----------------------------------------------
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.1, momentum=0.9, weight_decay=0.0005)

    sgdr = CosineAnnealingLR_with_Restart(optimizer,
                                          T_max=config.cycle_inter,
                                          T_mult=1,
                                          model=net,
                                          out_dir='../input/',
                                          take_snapshot=False,
                                          eta_min=1e-3)
    tprs = {"TPR@FPR=10E-2": 0.0, "TPR@FPR=10E-3": 0.0, "TPR@FPR=10E-4": 0.0}
    global_min_acer = 1.0
    id_best = 0
    for cycle_index in range(config.cycle_num):
        print('cycle index: ' + str(cycle_index))
        min_acer = 1.0

        for epoch in range(0, config.cycle_inter):
            sgdr.step()
            lr = optimizer.param_groups[0]['lr']
            print('lr : {:.4f}'.format(lr))

            sum_train_loss = np.zeros(6,np.float32)
            sum_e = 0
            optimizer.zero_grad()

            for input, truth in train_loader:
                iter = i + start_iter
                # one iteration update  -------------
                net.train()
                input = input.cuda()
                truth = truth.cuda()

                logit, ft, _ = net.forward(input)
                truth = truth.view(logit.shape[0])
                if config.criterion == 'triplet':
                    # print(criterion)
                    loss_ce = criterion['ce'](logit, truth)
                    loss_triplet = criterion['triplet'](truth, ft)
                    loss = loss_ce + loss_triplet

                elif config.criterion == 'ce':
                    loss  = criterion['ce'](logit, truth)
                elif config.criterion == 'crl':
                    loss_m  = criterion['crl'](logit, truth)
                    loss_c = criterion['focal'](logit, truth)        ### compute Focal Loss
                    # measure accuracy and record loss
                    # loss = (1-lalpha)*loss_c + lalpha*loss_m            ### compute CRL Loss
                    loss = loss_c

                precision,_ = metric(logit, truth)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # print statistics  ------------
                if config.criterion == 'triplet':
                    batch_loss[:3] = np.array(( loss_ce.item(), loss_triplet.item(), precision.item(),))
                elif config.criterion in ['ce', 'crl']:
                    batch_loss[:3] = np.array((loss.item(), 0,  precision.item()))
                sum_e += 1
                if iter%iter_smooth == 0:
                    train_loss = sum_train_loss/sum_e
                    sum_e = 0

                i = i + 1

            if epoch >= config.cycle_inter // 2:
            # if 1:
                net.eval()
                valid_loss, _, tprs = do_valid_test(net, valid_loader, criterion['ce'])
                net.train()

                if valid_loss[1] < min_acer and epoch > 0:
                    min_acer = valid_loss[1]
                    ckpt_name = out_dir + '/checkpoint/Cycle_' + str(cycle_index) + '_min_acer_model.pth'
                    torch.save(net.state_dict(), ckpt_name)
                    log.write('save cycle ' + str(cycle_index) + ' min acer model: ' + str(min_acer) + '\n')

                if valid_loss[1] < global_min_acer and epoch > 0:
                    global_min_acer = valid_loss[1]
                    ckpt_name = out_dir + '/checkpoint/global_min_acer_model_{}.pth'.format(id_best)
                    torch.save(net.state_dict(), ckpt_name)
                    log.write('save global min acer model: ' + str(min_acer) + '\n')
                    id_best += 1

            asterisk = ' '
            log.write(config.model_name+' Cycle %d: %0.4f %5.1f %3.1f | %0.6f  %0.6f  %0.3f  %0.4f  %0.4f  %0.4f  %s| %0.6f  %0.6f %0.6f |%s \n' % (
                cycle_index, lr, iter, epoch,
                valid_loss[0], valid_loss[1], valid_loss[2], tprs["TPR@FPR=10E-2"], tprs["TPR@FPR=10E-3"], tprs["TPR@FPR=10E-4"], asterisk,
                batch_loss[0], batch_loss[1], batch_loss[2],
                time_to_str((timer() - start), 'min')))

        ckpt_name = out_dir + '/checkpoint/Cycle_' + str(cycle_index) + '_final_model.pth'
        torch.save(net.state_dict(), ckpt_name)
        log.write('save cycle ' + str(cycle_index) + ' final model \n')

def run_test(config, dir):
    config.model_name = config.model + '_' + config.image_mode + '_' + str(config.image_size)
    out_dir = config.saved_path
    out_dir = os.path.join(out_dir,config.model_name)
    initial_checkpoint = config.pretrained_model

    ## net ---------------------------------------
    # from convert import *
    # net = get_model(model_name=config.model, num_class=2)
    # net = torch.nn.DataParallel(net)
    net, initial_checkpoint = convert(config.pretrained_model, model_name=config.model)
    net =  net.cuda()
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print('total_params: ', pytorch_total_params)
    # exit(0)
    if initial_checkpoint is not None:
        save_dir = os.path.join(out_dir + '/checkpoint', dir, initial_checkpoint)
        initial_checkpoint = os.path.join(out_dir +'/checkpoint',initial_checkpoint)
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        print('ckpt: ', initial_checkpoint)
        ckpt = torch.load(config.pretrained_model, map_location=lambda storage, loc: storage)
        net = load_checkpoint(net, ckpt)
        if not os.path.exists(os.path.join(out_dir + '/checkpoint', dir)):
            os.makedirs(os.path.join(out_dir + '/checkpoint', dir))
    
    mem_params = sum([param.nelement()*param.element_size() for param in net.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in net.buffers()])
    mem = mem_params + mem_bufs # in bytes
    print('mem on GPU', mem)

    criterion = softmax_cross_entropy_criterion
    net.eval()
    if not config.phase_test:
        valid_dataset = FDDataset(mode = 'val', modality=config.image_mode,image_size=config.image_size,
                                fold_index=config.train_fold_index)
        valid_loader  = DataLoader( valid_dataset,
                                    shuffle=False,
                                    batch_size=config.batch_size,
                                    drop_last=False,
                                    num_workers=8)
        valid_loss,out,tprs = do_valid_test(net, valid_loader, criterion)
    else:
        test_dataset = FDDataset(mode='test', modality=config.image_mode, image_size=config.image_size,
                                fold_index=config.train_fold_index, cross_test=config.cross_test)
        test_loader  = DataLoader( test_dataset,
                                    shuffle=False,
                                    batch_size=config.batch_size,
                                    drop_last=False,
                                    num_workers=8)
        valid_loss,out,tprs = do_valid_test(net, test_loader, criterion)
    print('%0.6f  %0.6f  %0.3f  (%0.3f) \n' % (valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3]))

    print('infer!!!!!!!!!')
    # out = infer_test(net, test_loader)
    print('done')

    # submission(out,save_dir+'_noTTA.txt', mode='test')

def main(config):
    if config.mode == 'train':
        run_train(config)

    if config.mode == 'infer_test':
        # config.pretrained_model = r'global_min_acer_model.pth'
        run_test(config, dir='global_test_36_TTA')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_fold_index', type=int, default = -1)
    parser.add_argument('--model', type=str, default='baseline')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--image_mode', type=str, default='fusion')
    parser.add_argument('--criterion', type=str, default='ce')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--cycle_num', type=int, default=10)
    parser.add_argument('--cycle_inter', type=int, default=50)
    parser.add_argument('--saved-path', type=str, default='./models')
    parser.add_argument('--mode', type=str, default='train', choices=['train','infer_test'])
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--phase-test', action='store_true')
    parser.add_argument('--cross-test', action='store_true')
    config = parser.parse_args()
    print(config)
    main(config)