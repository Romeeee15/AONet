import torch
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
import cv2
import logging
import os
from AONet import AONet
import utils
import numpy as np
from options.test_options import TestOptions
import natsort
import h5_dataloader as BatchDataReader

from pathlib import Path
import re
import glob
def increment_path(path, exist_ok=False, sep='', mkdir=True):
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir_ = path if path.suffix == '' else path.parent  # directory
    if not dir_.exists() and mkdir:
        dir_.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def test_net(net,device):
    data_clip = "Path to clip.json"
    dataset = BatchDataReader.CubeDataset(opt.dataroot, opt.val_ids, opt.modality_filename, opt.label_filename, data_clip=data_clip, is_dataaug=False)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    test_visuals = os.path.join('./test_visuals/', f'{opt.method}_{opt.modality_filename[0]}')
    test_visuals = increment_path(test_visuals)
    
    BGR=np.zeros((opt.data_size[1],opt.data_size[2],3))

    print(test_visuals)
    
    test_mIou = 0.
    with torch.no_grad():
        net.eval()
        #test set
        pbar = tqdm(enumerate(BackgroundGenerator(test_loader)), total=len(test_loader))
        for itr, (test_images, test_annotations, cubename) in pbar:
            result = np.zeros((test_images.shape[0], test_images.shape[-1], test_images.shape[-1]))
            test_images = test_images.to(device=device, dtype=torch.float32)
            pred = net(test_images)
            pred_argmax = torch.argmax(pred, dim=1)
            pred_argmax = pred_argmax.cpu().detach().numpy()

            result[:, :, :] = pred_argmax
            test_anno = test_annotations.squeeze().cpu().detach().numpy()
            test_anno = np.expand_dims(test_anno, axis=0)
            sample_Iou = utils.cal_miou(result, test_anno)
            test_mIou += sample_Iou
            pred_softmax = torch.nn.functional.softmax(pred, dim=1)
            pred_softmax = pred_softmax.cpu().detach().numpy()
            result = result.astype(np.uint8)
            im_color = (result[0] * 50)
            im_color = cv2.applyColorMap(cv2.convertScaleAbs(im_color, alpha=1), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(test_visuals, f'{cubename[0]}.bmp'), im_color)


        
        print(f'test_mIou: {test_mIou/len(test_loader)}')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    opt = TestOptions().parse()
    if opt.method=='AONet':
        net = AONet(opt.slicelen, opt.n_classes) 
    net = net.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bestmodelpath = r'Path to best.pth'
    
    print(bestmodelpath)
    net.load_state_dict(
        torch.load(bestmodelpath, map_location=device)
    )
    net.to(device=device)
    try:
        test_net(net=net,device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)