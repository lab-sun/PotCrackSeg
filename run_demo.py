# By Yuxiang Sun, Dec. 14, 2020
# Email: sun.yuxiang@outlook.com


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
from sklearn.metrics import confusion_matrix
from util.util import compute_results, visualize
from scipy.io import savemat 
from torch.utils.tensorboard import SummaryWriter

from model import PotCrackSeg 

from util.lr_policy import WarmUpPolyLR
from util.init_func import init_weight, group_weight
from config import config
# from thop import profile
from PIL import Image 

#############################################################################################
parser = argparse.ArgumentParser(description='Test with pytorch')
#############################################################################################
parser.add_argument('--model_name', '-m', type=str, default='PotCrackSeg')  # DRCNet_RDe_b3V3, DRCNet_RDe_b4V3, DRCNet_RDe_b5V3 
parser.add_argument('--weight_name', '-w', type=str, default='PotCrackSeg-5B') # DRCNet_RDe_b3V3, DRCNet_RDe_b4V3, DRCNet_RDe_b5V3 
parser.add_argument('--backbone', '-bac', type=str, default='PotCrackSeg-5B')  # mit_3, mit_4, mit_5
parser.add_argument('--file_name', '-f', type=str, default='final.pth')
parser.add_argument('--dataset_split', '-d', type=str, default='test') # normal_test, abnormal_test, urban_test,rural_test
parser.add_argument('--gpu', '-g', type=int, default=1)
#############################################################################################
parser.add_argument('--img_height', '-ih', type=int, default=288) 
parser.add_argument('--img_width', '-iw', type=int, default=512)  
parser.add_argument('--num_workers', '-j', type=int, default=16)
parser.add_argument('--n_class', '-nc', type=int, default=3)
parser.add_argument('--data_dir', '-dr', type=str, default='./NPO++/')
parser.add_argument('--model_dir', '-wd', type=str, default='./weights_backup/')
args = parser.parse_args()
#############################################################################################

def get_palette():
    unlabelled = [0,0,0]
    potholes        = [153,0,0]
    cracks     = [0,153,0]
    palette    = np.array([unlabelled,potholes, cracks])
    return palette


def visualize_feature(image_name,predictions,type = "rgb"):
    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()
        pred = np.transpose(pred, (1,2,0))
        img = Image.fromarray(np.uint8(pred*255))
        img.save('runs/Pred_' +type+ image_name[i] + '.png')


def visualize2(image_name, predictions, weight_name):
    palette = get_palette()
    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(0, len(palette)): # fix the mistake from the MFNet code on Dec.27, 2019
            img[pred == cid] = palette[cid]
        img = Image.fromarray(np.uint8(img))
        img.save('runs/Pred_' + weight_name + '_' + image_name[i] + '.png')



if __name__ == '__main__':
  
    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')

    # prepare save direcotry
    if os.path.exists("./runs"):
        print("previous \"./runs\" folder exist, will delete this folder")
        shutil.rmtree("./runs")
    os.makedirs("./runs")
    os.chmod("./runs", stat.S_IRWXO)  # allow the folder created by docker read, written, and execuated by local machine
    model_dir = os.path.join(args.model_dir, args.weight_name)
    if os.path.exists(model_dir) is False:
        sys.exit("the %s does not exit." %(model_dir))
    model_file = os.path.join(model_dir, args.file_name)
    if os.path.exists(model_file) is True:
        print('use the final model file.')
    else:
        sys.exit('no model file found.') 
    print('testing %s: %s on GPU #%d with pytorch' % (args.model_name, args.weight_name, args.gpu))
    
    conf_total = np.zeros((args.n_class, args.n_class))
    model = eval(args.model_name)(cfg = config ,n_class=args.n_class, encoder_name=args.backbone)
    if args.gpu >= 0: model.cuda(args.gpu)
    print('loading model file %s... ' % model_file)
    pretrained_weight = torch.load(model_file, map_location = lambda storage, loc: storage.cuda(args.gpu))
    own_state = model.state_dict()
    for name, param in pretrained_weight.items():
        own_state[name].copy_(param)  
    print('done!')

    batch_size = 1
    test_dataset  = MY_dataset(data_dir=args.data_dir, split=args.dataset_split, input_h=args.img_height, input_w=args.img_width)
    test_loader  = DataLoader(
        dataset     = test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    ave_time_cost = 0.0

    model.eval()
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(test_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            # flop,params = profile(model,inputs=(images,))
            # print(flop)
            # print(params)
            torch.cuda.synchronize()
            start_time = time.time()
            
            #rgb_result = model(images) #   For Single modal 

            rgb_predict, rgb_comple, rgb_fusion, depth_predict, depth_comple, depth_fusion, logits= model(images)
            torch.cuda.synchronize()
            end_time = time.time()

            # visualize_feature(image_name=names,predictions=rgb_predict,type="rgb_predict")
            # visualize_feature(image_name=names,predictions=rgb_comple,type="rgb_comple")
            # visualize_feature(image_name=names,predictions=rgb_fusion,type="rgb_fusion")
            # visualize_feature(image_name=names,predictions=depth_predict,type="depth_predict")
            # visualize_feature(image_name=names,predictions=depth_comple,type="depth_comple")
            # visualize_feature(image_name=names,predictions=depth_fusion,type="depth_fusion")
            visualize2(image_name=names, predictions=rgb_predict.argmax(1), weight_name="rgb_predict")
            visualize2(image_name=names, predictions=rgb_comple.argmax(1), weight_name="rgb_comple")
            visualize2(image_name=names, predictions=rgb_fusion.argmax(1), weight_name="rgb_fusion")
            visualize2(image_name=names, predictions=depth_predict.argmax(1), weight_name="depth_predict")
            visualize2(image_name=names, predictions=depth_comple.argmax(1), weight_name="depth_comple")
            visualize2(image_name=names, predictions=depth_fusion.argmax(1), weight_name="depth_fusion")

            
            if it>=5: # # ignore the first 5 frames
                ave_time_cost += (end_time-start_time)
            # convert tensor to numpy 1d array
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logits.argmax(1).cpu().numpy().squeeze().flatten() # prediction and label are both 1-d array, size: minibatch*640*480
            # generate confusion matrix frame-by-frame
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0,1,2]) # conf is an n_class*n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf
            # save demo images
            visualize(image_name=names, predictions=logits.argmax(1), weight_name=args.weight_name)
            print("%s, %s, frame %d/%d, %s, time cost: %.2f ms, demo result saved."
                  %(args.model_name, args.weight_name, it+1, len(test_loader), names, (end_time-start_time)*1000))
 
    precision_per_class, recall_per_class, iou_per_class,F1_per_class = compute_results(conf_total)
    #precision, recall, IoU,F1 = compute_results(conf_total)
    conf_total_matfile = os.path.join("./runs", 'conf_'+args.weight_name+'.mat')
    savemat(conf_total_matfile,  {'conf': conf_total}) # 'conf' is the variable name when loaded in Matlab
 
    print('\n###########################################################################')
    print('\n%s: %s test results (with batch size %d) on %s using %s:' %(args.model_name, args.weight_name, batch_size, datetime.date.today(), torch.cuda.get_device_name(args.gpu))) 
    print('\n* the tested dataset name: %s' % args.dataset_split)
    print('* the tested image count: %d' % len(test_loader))
    print('* the tested image size: %d*%d' %(args.img_height, args.img_width)) 
    print('* the weight name: %s' %args.weight_name) 
    print('* the file name: %s' %args.file_name) 
    print("* iou per class: \n    unlabeled: %.1f, pothole: %.1f, crack: %.1f" \
          %(iou_per_class[0]*100, iou_per_class[1]*100, iou_per_class[2]*100)) 
    print("* recall per class: \n    unlabeled: %.1f, pothole: %.1f, crack: %.1f" \
          %(recall_per_class[0]*100, recall_per_class[1]*100, recall_per_class[2]*100))
    print("* pre per class: \n    unlabeled: %.1f, pothole: %.1f, crack: %.1f" \
          %(precision_per_class[0]*100, precision_per_class[1]*100, precision_per_class[2]*100)) 
    print("* F1 per class: \n    unlabeled: %.1f, pothole: %.1f, crack: %.1f" \
          %(F1_per_class[0]*100, F1_per_class[1]*100, F1_per_class[2]*100)) 

    print("\n* average values (np.mean(x)): \n iou: %.3f, recall: %.3f, pre: %.3f, F1: %.3f" \
          %(iou_per_class[1:].mean()*100,recall_per_class[1:].mean()*100, precision_per_class[1:].mean()*100,F1_per_class[1:].mean()*100))
    print("* average values (np.mean(np.nan_to_num(x))): \n iou: %.1f, recall: %.1f, pre: %.1f, F1: %.1f" \
          %(np.mean(np.nan_to_num(iou_per_class[1:]))*100, np.mean(np.nan_to_num(recall_per_class[1:]))*100, np.mean(np.nan_to_num(precision_per_class[1:]))*100, np.mean(np.nan_to_num(F1_per_class[1:]))*100))


    print('\n* the average time cost per frame (with batch size %d): %.2f ms, namely, the inference speed is %.2f fps' %(batch_size, ave_time_cost*1000/(len(test_loader)-5), 1.0/(ave_time_cost/(len(test_loader)-5)))) # ignore the first 10 frames
    print('\n###########################################################################')