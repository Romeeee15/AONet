import torch
import torch.nn as nn
import numpy as np
import logging
import os
from AONet import AONet
import utils
from myutils import mkdir_p, Logger
from datetime import datetime
import shutil
import natsort
from options.train_options import TrainOptions
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
import h5_dataloader as BatchDataReader
import scipy.io as io
import cv2
from torch.utils.data import DataLoader
torch.backends.cudnn.benchmark = False
import warnings
warnings.filterwarnings("ignore")

def train_net(net, device):
    #train setting
    num_workers = opt.num_workers
    val_num = opt.val_ids[1] - opt.val_ids[0]
    DATA_SIZE = opt.data_size
    
    best_model_save_path = os.path.join(opt.saveroot, f'{opt.method}/best_model/{opt.modality_filename[0]}')
    mkdir_p(best_model_save_path)

    logger = Logger(os.path.join(opt.saveroot, f'{opt.method}/best_model/{opt.modality_filename[0]}/train_process.txt'), title=opt.method)
    logger.set_names(["Epoch-Num", 'Train-Loss', 'Train-mIoU', 'Train-mDSC', 'Test-Loss', 'Test-mIoU', 'Test-mDSC'])
    
    # Read Data
    printf("Start Setup dataset reader")
    if opt.dataset == 'OCTA':
        printf("OCTA")
        data_clip = r"Path to clip.json"
        train_dataset = BatchDataReader.CubeDataset(opt.dataroot, opt.train_ids, opt.modality_filename, opt.label_filename, data_clip=data_clip, is_dataaug=True)
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=False, num_workers=num_workers)
        valid_dataset = BatchDataReader.CubeDataset(opt.dataroot, opt.val_ids, opt.modality_filename, opt.label_filename, data_clip=data_clip, is_dataaug=False)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, pin_memory=False, num_workers=num_workers)
        
    # Setting Optimizer
    if opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), opt.lr, momentum=0.9, weight_decay=1e-6)
    elif opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), opt.lr, betas=(0.9, 0.99))
    elif opt.optimizer == 'RMS':
        optimizer = torch.optim.RMSprop(net.parameters(), opt.lr, weight_decay=1e-8)
    
    
    #Setting Loss
    Loss_CE = nn.CrossEntropyLoss()
    Loss_DSC= utils.DiceLoss()
    
    #Start train
    best_valid_miou=0
    save_loss=[]
    for epoch in range(1, opt.num_epochs + 1):
        train_loss = 0.
        train_miou_sum = 0.
        train_dice_sum = 0.
        train_num = len(train_loader)
        net.train()
        pbar = tqdm(enumerate(BackgroundGenerator(train_loader)), total=len(train_loader))
        for itr, (train_images, train_annotations, cubename) in pbar:
            result = np.zeros((train_images.shape[0], train_images.shape[-1], train_images.shape[-1]))
            train_images_patch =train_images.to(device=device, dtype=torch.float32)
            train_annotations_patch = train_annotations.to(device=device, dtype=torch.long).squeeze(1)
            pred = net(train_images_patch)
            loss = Loss_CE(pred, train_annotations_patch) + 0.6*Loss_DSC(pred, train_annotations_patch)

                    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            save_loss.append(loss.item())
            train_loss += loss.item()

            pred_argmax = torch.argmax(pred, dim=1)
            result[:, :, :] = pred_argmax.cpu().detach().numpy()
            
            train_anno = np.squeeze(train_annotations).cpu().detach().numpy()
            train_miou_sum += utils.cal_miou(result, train_anno)
            train_dice_sum += utils.cal_Dice(result, train_anno)
        train_miou = train_miou_sum / train_num
        train_dice = train_dice_sum / train_num
        train_loss = train_loss / train_num

        #Start Val
        with torch.no_grad():

            val_miou_sum = 0
            val_dice_sum = 0
            val_loss = 0.
            net.eval()
            pbar = tqdm(enumerate(BackgroundGenerator(valid_loader)), total=len(valid_loader))
            for itr, (test_images, test_annotations, cubename) in pbar:
                result = np.zeros((test_images.shape[0], test_images.shape[-1], test_images.shape[-1]))
                test_images = test_images.to(device=device, dtype=torch.float32)
                test_annotations = test_annotations.to(device=device, dtype=torch.long).squeeze(1)
                pred = net(test_images)

                
                
                loss = Loss_CE(pred, test_annotations) + 0.6 * Loss_DSC(pred, test_annotations)

                val_loss += loss.item()
                pred_argmax = torch.argmax(pred, dim=1)
                result[:, :, :] = pred_argmax.cpu().detach().numpy()
                test_annotations = np.squeeze(test_annotations, axis=1).cpu().detach().numpy()
                val_miou_sum += utils.cal_miou(result, test_annotations)
                val_dice_sum += utils.cal_Dice(result, test_annotations)
            val_miou = val_miou_sum/val_num
            val_dice = val_dice_sum/val_num
            val_loss = val_loss/val_num
            printf("Step:{}, Valid_mIoU:{}".format(epoch,val_miou))

            
            if val_miou > best_valid_miou:
                temp = '{:.6f}'.format(val_miou)
                try:
                    os.mkdir(os.path.join(best_model_save_path, temp))
                except:
                    pass
                # temp2= f'{epoch}.pth'                
                torch.save(net.state_dict(),os.path.join(best_model_save_path, temp, f'{epoch}.pth'))
                
                model_names = natsort.natsorted(os.listdir(best_model_save_path))

                if epoch < 100:
                    if len(model_names) == 4:
                        shutil.rmtree(os.path.join(best_model_save_path,model_names[0]))
                elif epoch < 200:
                    if len(model_names) == 5:
                        shutil.rmtree(os.path.join(best_model_save_path,model_names[1]))   
                elif epoch < 300:
                    if len(model_names) == 6:
                        shutil.rmtree(os.path.join(best_model_save_path,model_names[2]))
                
                best_valid_miou = val_miou    
                

        printf(f'Epoch {epoch}, Train loss: {train_loss:.6f}, Train mIoU: {train_miou:.6f}, Train mDSC: {train_dice:.6f}, Val loss: {val_loss:.6f}, Val mIoU: {val_miou} (max :{best_valid_miou}), Val mDSC: {val_dice:.6f}')
    logger.close()
    printf(f"++++++++" * 2 + "Final results" + "++++++++" * 2)
    printf(f'Best val mIoU: {best_valid_miou}')

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    
    opt = TrainOptions().parse()
    print(opt.method)
    time_stamp = datetime.now()
    opt.saveroot = os.path.join(opt.saveroot,str(time_stamp))
    best_model_save_path = os.path.join(opt.saveroot, f'{opt.method}/best_model/{opt.modality_filename[0]}')
    mkdir_p(best_model_save_path)
    
    screen_logger = logging.getLogger("Model")
    screen_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(os.path.join(opt.saveroot, f"{opt.method}/best_model/{opt.modality_filename[0]}/screen_logger.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    screen_logger.addHandler(file_handler)

    def printf(str):
        screen_logger.info(str)
        print(f'**{str}**')

    #setting GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    printf(opt.opt_msg)
    printf(f'Using device {device}')
    
    if opt.dataset == 'OCTA':
        c_input = 320   
        if opt.method=='AONet':
            net = AONet(c_input, 5)
        else:
            raise NotImplementedError(f'{opt.method} not implement in this version.') 
    else:
        raise NotImplementedError(f"{opt.dataset} has not be impleted")
    
    
    if len(opt.gpu_ids) == 1:
        net = net.cuda()
    else:
        device_ids = opt.gpu_ids.split(',')
        device_ids = [int(i) for i in device_ids]
        net=torch.nn.DataParallel(net, device_ids).cuda()

    if opt.load:
        net.load_state_dict(
            torch.load(opt.load, map_location=device)
        )
        printf(f'Model loaded from {opt.load}')
        
        
    train_net(net=net, device=device)



