import os, argparse, time, datetime, stat, shutil,sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from util.MY_dataset import MY_dataset
from util.augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise
from util.util import compute_results
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from model import PotCrackSeg
from util.lr_policy import WarmUpPolyLR
from util.init_func import init_weight, group_weight
from config import config

#############################################################################################
parser = argparse.ArgumentParser(description='Train with pytorch')
############################################################################################# 
parser.add_argument('--model_name', '-m', type=str, default='PotCrackSeg')
parser.add_argument('--batch_size', '-b', type=int, default=2) 
parser.add_argument('--lr_start', '-ls', type=float, default=6e-5)
parser.add_argument('--gpu', '-g', type=int, default=0)
#############################################################################################
parser.add_argument('--lr_decay', '-ld', type=float, default=0.95)
parser.add_argument('--epoch_max', '-em', type=int, default=500) # please stop training mannully 
parser.add_argument('--epoch_from', '-ef', type=int, default=0) 
parser.add_argument('--num_workers', '-j', type=int, default=8)
parser.add_argument('--n_class', '-nc', type=int, default=3)
parser.add_argument('--data_dir', '-dr', type=str, default='./NPO++/')
parser.add_argument('--pre_weight', '-prw', type=str, default='/pretrained/mit_b2.pth')
parser.add_argument('--backbone', '-bac', type=str, default='PotCrackSeg-2B')
parser.add_argument('--model_dir', '-wd', type=str, default='./weights_backup/')
# parser.add_argument('--weight_name', '-w', type=str, default='DRCNet_0DRC_RDe_b0') # RTFNet_152, RTFNet_50, please change the number of layers in the network file
# parser.add_argument('--file_name', '-f', type=str, default='109.pth')
args = parser.parse_args()
#############################################################################################

augmentation_methods = [
    RandomFlip(prob=0.5),
    RandomCrop(crop_rate=0.1, prob=1.0),
    # RandomCropOut(crop_rate=0.2, prob=1.0),
    # RandomBrightness(bright_range=0.15, prob=0.9),
    # RandomNoise(noise_range=5, prob=0.9),
]

def fusion_loss(rgb_predict, rgb_comple,depth_predict, depth_comple,label):
    
    feature_map_B, feature_map_C, feature_map_W, feature_map_H = rgb_predict.size()
    label_B, label_W, label_H = label.size()

    if feature_map_W != label_W:
        label = torch.cuda.FloatTensor(label.unsqueeze(1).cpu().numpy())
        label = F.interpolate(label,[feature_map_W,feature_map_H],mode="nearest")
        label = torch.cuda.LongTensor(label.squeeze(1).cpu().numpy())

    loss_pr_rgb_seg = F.cross_entropy(rgb_predict, label)    
    rgb_predict = rgb_predict.detach()
    rgb_predict=rgb_predict.argmax(1)
    rgb_predict.eq_(label)
    rgb_predict=rgb_predict.clone().detach_().requires_grad_(False)
    add_map_rgb = (1-rgb_predict)*label
    add_map_rgb=add_map_rgb.clone().detach_().requires_grad_(False)
    loss_add_rgb = F.cross_entropy(rgb_comple,add_map_rgb)

    loss_pr_depth_seg = F.cross_entropy(depth_predict, label)  
    depth_predict = depth_predict.detach()
    depth_predict=depth_predict.argmax(1)
    depth_predict.eq_(label)
    depth_predict=depth_predict.clone().detach_().requires_grad_(False)
    add_map_depth = (1-depth_predict)*label
    add_map_depth=add_map_depth.clone().detach_().requires_grad_(False)
    loss_add_depth = F.cross_entropy(depth_comple,add_map_depth)

    loss = loss_pr_rgb_seg+loss_add_rgb+loss_pr_depth_seg+loss_add_depth
    return loss,add_map_rgb,add_map_depth



def train(epo, model, train_loader, optimizer):
    model.train()
    loss_sum = 0

    loss_seg_sum = 0
    for it, (images, labels, names) in enumerate(train_loader):
        images = Variable(images).cuda(args.gpu)
        labels = Variable(labels).cuda(args.gpu)

        start_t = time.time() # time.time() returns the current time
        optimizer.zero_grad()
        rgb_predict, rgb_comple, rgb_fusion, depth_predict, depth_comple, depth_fusion, logits= model(images)

        loss1,add_map_rgb,add_map_depth = fusion_loss(rgb_predict, rgb_comple,depth_predict, depth_comple,labels)
        #loss2 = F.cross_entropy(rgb_fusion, labels)
        #loss3 = F.cross_entropy(depth_fusion, labels)
        loss4 = F.cross_entropy(logits, labels)

        #loss = 0.5*loss1+loss2+loss3+loss4
        loss = 0.5*loss1+loss4

        loss.backward()
        optimizer.step()

        loss_sum = loss_sum+loss
        loss_seg_sum = loss_seg_sum+loss4

        current_idx = (epo- 0) * config.niters_per_epoch + it
        lr = lr_policy.get_lr(current_idx)

        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr

        lr_this_epo=0
        for param_group in optimizer.param_groups:
            lr_this_epo = param_group['lr']

        print('Train: %s, epo %s/%s, iter %s/%s, lr %.8f, %.2f img/sec, loss %.4f, loss_average %.4f, loss_seg_average %.4f, time %s' \
            % (args.model_name, epo, args.epoch_max, it+1, len(train_loader), lr_this_epo, len(names)/(time.time()-start_t), float(loss), float(loss_sum/(it+1)), float(loss_seg_sum/(it+1)),
              datetime.datetime.now().replace(microsecond=0)-start_datetime))
        if accIter['train'] % 1 == 0:
            writer.add_scalar('Train/loss', loss, accIter['train'])
        view_figure = True # note that I have not colorized the GT and predictions here
        if accIter['train'] % 1000 == 0:
            if view_figure:
                input_rgb_images = vutils.make_grid(images[:,:3], nrow=8, padding=10) # can only display 3-channel images, so images[:,:3]
                writer.add_image('Train/input_rgb_images', input_rgb_images, accIter['train'])
                scale = max(1, 255//args.n_class) # label (0,1,2..) is invisable, multiply a constant for visualization
                groundtruth_tensor = labels.unsqueeze(1) * scale  # mini_batch*480*640 -> mini_batch*1*480*640
                groundtruth_tensor = torch.cat((groundtruth_tensor, groundtruth_tensor, groundtruth_tensor), 1)  # change to 3-channel for visualization
                groudtruth_images = vutils.make_grid(groundtruth_tensor, nrow=8, padding=10)
                writer.add_image('Train/groudtruth_images', groudtruth_images, accIter['train'])
                predicted_tensor = logits.argmax(1).unsqueeze(1) * scale # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                predicted_tensor = torch.cat((predicted_tensor, predicted_tensor, predicted_tensor),1) # change to 3-channel for visualization, mini_batch*1*480*640
                predicted_images = vutils.make_grid(predicted_tensor, nrow=8, padding=10)
                writer.add_image('Train/predicted_images', predicted_images, accIter['train'])

                predicted_tensor_rgb_1 = rgb_predict.argmax(1).unsqueeze(1) * scale # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                predicted_tensor_rgb_1 = torch.cat((predicted_tensor_rgb_1, predicted_tensor_rgb_1, predicted_tensor_rgb_1),1) # change to 3-channel for visualization, mini_batch*1*480*640
                predicted_images_rgb_1 = vutils.make_grid(predicted_tensor_rgb_1, nrow=8, padding=10)
                writer.add_image('Train/predicted_images_rgb_1', predicted_images_rgb_1, accIter['train'])

                predicted_tensor_depth_1 = depth_predict.argmax(1).unsqueeze(1) * scale # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                predicted_tensor_depth_1 = torch.cat((predicted_tensor_depth_1, predicted_tensor_depth_1, predicted_tensor_depth_1),1) # change to 3-channel for visualization, mini_batch*1*480*640
                predicted_images_depth_1 = vutils.make_grid(predicted_tensor_depth_1, nrow=8, padding=10)
                writer.add_image('Train/predicted_images_depth_1', predicted_images_depth_1, accIter['train'])

                add_map_rgb_1 = add_map_rgb.unsqueeze(1) * scale 
                predicted_tensor_need_rgb_1 = torch.cat((add_map_rgb_1, add_map_rgb_1, add_map_rgb_1), 1)  # change to 3-channel for visualization
                predicted_images_need_rgb_1 = vutils.make_grid(predicted_tensor_need_rgb_1, nrow=8, padding=10)
                writer.add_image('Train/predicted_images_need_rgb_1', predicted_images_need_rgb_1, accIter['train'])

                add_map_depth_1 = add_map_depth.unsqueeze(1) * scale 
                predicted_tensor_need_depth_1 = torch.cat((add_map_depth_1, add_map_depth_1, add_map_depth_1), 1)  # change to 3-channel for visualization
                predicted_images_need_depth_1 = vutils.make_grid(predicted_tensor_need_depth_1, nrow=8, padding=10)
                writer.add_image('Train/predicted_images_need_depth_1', predicted_images_need_depth_1, accIter['train'])

                predicted_tensor_complex_rgb_1 = rgb_comple.argmax(1).unsqueeze(1) * scale # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                predicted_tensor_complex_rgb_1 = torch.cat((predicted_tensor_complex_rgb_1, predicted_tensor_complex_rgb_1, predicted_tensor_complex_rgb_1),1) # change to 3-channel for visualization, mini_batch*1*480*640
                predicted_images_complex_rgb_1 = vutils.make_grid(predicted_tensor_complex_rgb_1, nrow=8, padding=10)
                writer.add_image('Train/predicted_images_complex_rgb_1', predicted_images_complex_rgb_1, accIter['train'])

                predicted_tensor_complex_depth_1 = depth_comple.argmax(1).unsqueeze(1) * scale # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                predicted_tensor_complex_depth_1 = torch.cat((predicted_tensor_complex_depth_1, predicted_tensor_complex_depth_1, predicted_tensor_complex_depth_1),1) # change to 3-channel for visualization, mini_batch*1*480*640
                predicted_images_complex_depth_1 = vutils.make_grid(predicted_tensor_complex_depth_1, nrow=8, padding=10)
                writer.add_image('Train/predicted_images_complex_depth_1', predicted_images_complex_depth_1, accIter['train'])
        accIter['train'] = accIter['train'] + 1

def validation(epo, model, val_loader): 
    model.eval()
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(val_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            start_t = time.time() # time.time() returns the current time
            rgb_predict, rgb_comple, rgb_fusion, depth_predict, depth_comple, depth_fusion, logits= model(images)
            loss = F.cross_entropy(logits, labels)  # Note that the cross_entropy function has already include the softmax function
            print('Val: %s, epo %s/%s, iter %s/%s, %.2f img/sec, loss %.4f, time %s' \
                  % (args.model_name, epo, args.epoch_max, it + 1, len(val_loader), len(names)/(time.time()-start_t), float(loss),
                    datetime.datetime.now().replace(microsecond=0)-start_datetime))
            if accIter['val'] % 1 == 0:
                writer.add_scalar('Validation/loss', loss, accIter['val'])
            view_figure = False  # note that I have not colorized the GT and predictions here
            if accIter['val'] % 1000 == 0:
                if view_figure:
                    input_rgb_images = vutils.make_grid(images[:, :3], nrow=8, padding=10)  # can only display 3-channel images, so images[:,:3]
                    writer.add_image('Validation/input_rgb_images', input_rgb_images, accIter['val'])
                    scale = max(1, 255 // args.n_class)  # label (0,1,2..) is invisable, multiply a constant for visualization
                    groundtruth_tensor = labels.unsqueeze(1) * scale  # mini_batch*480*640 -> mini_batch*1*480*640
                    groundtruth_tensor = torch.cat((groundtruth_tensor, groundtruth_tensor, groundtruth_tensor), 1)  # change to 3-channel for visualization
                    groudtruth_images = vutils.make_grid(groundtruth_tensor, nrow=8, padding=10)
                    writer.add_image('Validation/groudtruth_images', groudtruth_images, accIter['val'])
                    predicted_tensor = logits.argmax(1).unsqueeze(1)*scale  # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                    predicted_tensor = torch.cat((predicted_tensor, predicted_tensor, predicted_tensor), 1)  # change to 3-channel for visualization, mini_batch*1*480*640
                    predicted_images = vutils.make_grid(predicted_tensor, nrow=8, padding=10)
                    writer.add_image('Validation/predicted_images', predicted_images, accIter['val'])
            accIter['val'] += 1

def testing(epo, model, test_loader):
    model.eval()
    conf_total = np.zeros((args.n_class, args.n_class))
    label_list = ["unlabeled", "pothole", "crack"]
    testing_results_file = os.path.join(weight_dir, 'testing_results_file.txt')
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(test_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            rgb_predict, rgb_comple, rgb_fusion, depth_predict, depth_comple, depth_fusion, logits= model(images)
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logits.argmax(1).cpu().numpy().squeeze().flatten() # prediction and label are both 1-d array, size: minibatch*640*480
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0,1,2]) # conf is args.n_class*args.n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf
            print('Test: %s, epo %s/%s, iter %s/%s, time %s' % (args.model_name, epo, args.epoch_max, it+1, len(test_loader),
                 datetime.datetime.now().replace(microsecond=0)-start_datetime))
    precision, recall, IoU, F1 = compute_results(conf_total)
    writer.add_scalar('Test/average_recall', recall.mean(), epo)
    writer.add_scalar('Test/average_IoU', IoU.mean(), epo)
    writer.add_scalar('Test/average_precision',precision.mean(), epo)
    writer.add_scalar('Test/average_F1', F1.mean(), epo)
    for i in range(len(precision)):
        writer.add_scalar("Test(class)/precision_class_%s" % label_list[i], precision[i], epo)
        writer.add_scalar("Test(class)/recall_class_%s"% label_list[i], recall[i],epo)
        writer.add_scalar('Test(class)/Iou_%s'% label_list[i], IoU[i], epo)
        writer.add_scalar('Test(class)/F1_%s'% label_list[i], F1[i], epo)
    if epo==0:
        with open(testing_results_file, 'w') as f:
            f.write("# %s, initial lr: %s, batch size: %s, date: %s \n" %(args.model_name, args.lr_start, args.batch_size, datetime.date.today()))
            f.write("# epoch: unlabeled, car, person, bike, curve, car_stop, guardrail, color_cone, bump,  average(nan_to_num). (Pre %, Acc %, IoU %, F1 %)\n")
    with open(testing_results_file, 'a') as f:
        f.write(str(epo)+': ')
        for i in range(len(precision)):
            f.write('%0.4f, %0.4f, %0.4f, %0.4f ' % (100*precision[i], 100*recall[i], 100*IoU[i], 100*F1[i]))
        f.write('%0.4f, %0.4f, %0.4f, %0.4f\n' % (100*np.mean(np.nan_to_num(precision)), 100*np.mean(np.nan_to_num(recall)), 100*np.mean(np.nan_to_num(IoU)), 100*np.mean(np.nan_to_num(F1)) ))
        #f.write('%0.4f, %0.4f, %0.4f, %0.4f\n' % (100*np.mean(np.nan_to_num(recall)), 100*np.mean(np.nan_to_num(IoU), 100*np.mean(np.nan_to_num(precision)), ))))
    print('saving testing results.')
    with open(testing_results_file, "r") as file:
        writer.add_text('testing_results', file.read().replace('\n', '  \n'), epo)

if __name__ == '__main__':
   
    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')

    config.pretrained_model = config.root_dir + args.pre_weight

    model = eval(args.model_name)(cfg = config ,n_class=args.n_class, encoder_name=args.backbone)

    base_lr = args.lr_start
    
    if args.gpu >= 0: model.cuda(args.gpu)
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_start, momentum=0.9, weight_decay=0.0005)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay, last_epoch=-1)
    params_list = []
    params_list = group_weight(params_list, model, nn.BatchNorm2d, base_lr)
    optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)

    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    # preparing folders
    if os.path.exists("./PotCrackSeg"):
        shutil.rmtree("./PotCrackSeg")
    weight_dir = os.path.join("./PotCrackSeg", args.model_name)
    os.makedirs(weight_dir)
    os.chmod(weight_dir, stat.S_IRWXO)  # allow the folder created by docker read, written, and execuated by local machine
 
    writer = SummaryWriter("./PotCrackSeg/tensorboard_log")
    os.chmod("./PotCrackSeg/tensorboard_log", stat.S_IRWXO)  # allow the folder created by docker read, written, and execuated by local machine
    os.chmod("./PotCrackSeg", stat.S_IRWXO) 

    print('training %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    print('from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    print('weight will be saved in: %s' % weight_dir)

    train_dataset = MY_dataset(data_dir=args.data_dir, split='train', transform=augmentation_methods,input_h=288, input_w=512)
    val_dataset  = MY_dataset(data_dir=args.data_dir, split='validation',input_h=288, input_w=512)
    test_dataset = MY_dataset(data_dir=args.data_dir, split='test',input_h=288, input_w=512)

    train_loader  = DataLoader(
        dataset     = train_dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    val_loader  = DataLoader(
        dataset     = val_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    test_loader = DataLoader(
        dataset      = test_dataset,
        batch_size   = args.batch_size,
        shuffle      = False,
        num_workers  = args.num_workers,
        pin_memory   = True,
        drop_last    = False
    )
    start_datetime = datetime.datetime.now().replace(microsecond=0)
    accIter = {'train': 0, 'val': 0}
    for epo in range(args.epoch_from, args.epoch_max):
        print('\ntrain %s, epo #%s begin...' % (args.model_name, epo))
        #scheduler.step() # if using pytorch 0.4.1, please put this statement here 
        train(epo, model, train_loader, optimizer)
        validation(epo, model, val_loader)

        checkpoint_model_file = os.path.join(weight_dir, str(epo) + '.pth')
        print('saving check point %s: ' % checkpoint_model_file)
        torch.save(model.state_dict(), checkpoint_model_file)

        testing(epo, model, test_loader)
        #scheduler.step() # if using pytorch 1.1 or above, please put this statement here
