
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 10:50:25 2019
@author: Keshik
"""
import argparse
import torch
import numpy as np
from torchvision import transforms
import torchvision.models as  models
from torch.utils.data import DataLoader
from dataset import PascalVOC_Dataset,  CocoDetection, CutoutPIL
# from randaugment import RandAugment
import torch.optim as optim
from train import train_model, test
from attack import tkmlap
from baseline_attacks import baselineap
from utils import encode_labels, plot_history
import os
import torch.utils.model_zoo as model_zoo
import utils
from models.inception import Inception3
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import matplotlib.gridspec as gridspec
from randaugment import RandAugment

os.environ['TORCH_HOME'] = '.'
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def main(args):
    data_dir = args.data
    model_name = args.arch
    num = args.num
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    download_data = args.download_data
    save_results = args.save_results

    model_dir = os.path.join(args.results, args.arch)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'inception_v3': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
    }

    model_collections_dict = {
        "resnet18": models.resnet18(),
        "resnet34": models.resnet34(),
        "resnet50": models.resnet50(),
        "inception_v3": models.inception_v3()
    }

    # Initialize cuda parameters
    use_cuda = torch.cuda.is_available()
    np.random.seed(2019)
    torch.manual_seed(2019)
    device = torch.device("cuda" if use_cuda else "cpu")
    # device = torch.device("cpu")
    print("Available device = ", device)

    if model_name in ['resnet18', 'resnet34', 'resnet50', 'inception_v3']:
        model = model_collections_dict[model_name]
        model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        model.load_state_dict(model_zoo.load_url(model_urls[model_name]))
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, args.num_classes)
    else:
        model = Inception3(num_classes=args.num_classes)

    model.to(device)

    if args.opt == 'SGD':
        optimizer = optim.SGD([
            {'params': list(model.parameters())[:-1], 'lr': lr[0], 'momentum': 0.9},
            {'params': list(model.parameters())[-1], 'lr': lr[1], 'momentum': 0.9}
        ])
    elif args.opt == 'Adam':
        optimizer = optim.Adam([
            {'params': list(model.parameters())[:-1], 'lr': lr[0], 'momentum': 0.9},
            {'params': list(model.parameters())[-1], 'lr': lr[1], 'momentum': 0.9}
        ])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 12, eta_min=0, last_epoch=-1)

    # Imagnet values
    # mean=[0.457342265910642, 0.4387686270106377, 0.4073427106250871]
    # std=[0.26753769276329037, 0.2638145880487105, 0.2776826934044154]
    if args.normalize == 'mean_std':
        mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
        std = [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]
    elif args.normalize == 'boxmaxmin':
        if args.boxmax == 1 and args.boxmin == 0:
            mean = [0, 0, 0]
            std = [1.0, 1.0, 1.0]
        elif args.boxmax == -(args.boxmin):
            mean = [0.5, 0.5, 0.5]
            std = [0.5 / args.boxmax, 0.5 / args.boxmax, 0.5 / args.boxmax]
        else:
            return

    if args.dataset == 'COCO':
        # COCO DataLoader
        instances_path_val = os.path.join(args.data, 'annotations/instances_val2014.json')
        instances_path_train = os.path.join(args.data, 'annotations/instances_train2014.json')
        # data_path_val = args.data
        # data_path_train = args.data
        data_path_val = f'{args.data}/val2014'  # args.data
        data_path_train = f'{args.data}/train2014'  # args.data
        COCO_val_dataset = CocoDetection(data_path_val,
                                         instances_path_val,
                                         transforms.Compose([
                                             transforms.Resize((args.image_size, args.image_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std),
                                             # normalize, # no need, toTensor does normalization
                                         ]))
        COCO_train_dataset = CocoDetection(data_path_train,
                                           instances_path_train,
                                           transforms.Compose([
                                               transforms.Resize((args.image_size, args.image_size)),
                                               CutoutPIL(cutout_factor=0.5),
                                               RandAugment(),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=mean, std=std),
                                               # normalize,
                                           ]))
        print('Using COCO dataset')
        print("COCO len(val_dataset)): ", len(COCO_val_dataset))
        print("COCO len(train_dataset)): ", len(COCO_train_dataset))
        train_loader = torch.utils.data.DataLoader(
            COCO_train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        # valid_loader = torch.utils.data.DataLoader(
        #     COCO_val_dataset, batch_size=args.batch_size, shuffle=False,
        #     num_workers=args.workers, pin_memory=False)
        valid_loader = torch.utils.data.DataLoader(
            COCO_val_dataset, batch_size=args.batch_size,
            num_workers=args.workers)
    elif args.dataset == 'VOC':
        # Create VOC train dataloader
        transformations = transforms.Compose([transforms.Resize((300, 300)),

                                              transforms.RandomChoice([
                                                  transforms.ColorJitter(brightness=(0.80, 1.20)),
                                                  transforms.RandomGrayscale(p=0.25)
                                              ]),
                                              transforms.RandomHorizontalFlip(p=0.25),
                                              transforms.RandomRotation(25),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=mean, std=std),
                                              ])

        VOC_dataset_train = PascalVOC_Dataset(data_dir,
                                              year='2012',
                                              image_set='train',
                                              download=download_data,
                                              transform=transformations,
                                              target_transform=encode_labels)

        # VOC validation dataloader
        transformations_valid = transforms.Compose([transforms.Resize(330),
                                                    transforms.CenterCrop(300),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=mean, std=std),
                                                    ])

        VOC_dataset_valid = PascalVOC_Dataset(data_dir,
                                              year='2012',
                                              image_set='val',
                                              download=download_data,
                                              transform=transformations_valid,
                                              target_transform=encode_labels)
        train_loader = DataLoader(VOC_dataset_train, batch_size=batch_size, num_workers=4, shuffle=True)
        valid_loader = DataLoader(VOC_dataset_valid, batch_size=batch_size, num_workers=4)

        # VOC testing loader
        transformations_test = transforms.Compose([transforms.Resize(330),
                                                   transforms.FiveCrop(300),
                                                   transforms.Lambda(lambda crops: torch.stack(
                                                       [transforms.ToTensor()(crop) for crop in crops])),
                                                   transforms.Lambda(lambda crops: torch.stack(
                                                       [transforms.Normalize(mean=mean, std=std)(crop) for crop in
                                                        crops])),
                                                   ])

        dataset_test = PascalVOC_Dataset(data_dir,
                                         year='2012',
                                         image_set='val',
                                         download=download_data,
                                         transform=transformations_test,
                                         target_transform=encode_labels)

    # ---------------Test your model here---------------------------------------
    # Load the best weights before testing

    #---------------Test your model here---------------------------------------
    # Load the best weights before testing
    weights_file_path = os.path.join(model_dir, "model-{}.pth".format(num))
    if os.path.isfile(weights_file_path):
        print("Loading best weights")
        model.load_state_dict(torch.load(weights_file_path))

    # for ith, (data, GT) in tqdm(enumerate(valid_loader)):
    #     if ith in [5]:
    #         plt.imshow(data[0].numpy().transpose((1, 2, 0)) * 0.5 + 0.5)
    #
    #         # plt.savefig('debug.png')
    #         # img = Image.fromarray(data[0].cpu().detach().numpy().transpose((2, 1, 0)))
    #         # img.save('test.jpg')
    #         # plt.imshow(img)
    #         # cv2.imwrite("image", data[0].cpu().detach().numpy().transpose((2, 1, 0)))
    #         plt.show()



    return model, device, valid_loader

def plot(args, model, device, valid_loader):
    model.eval()




    target_attack_result = np.load('./result/{}/{}/{}/eps_{}/k_{}.npy'.format(args.dataset, args.label_difficult, 'none_target_attack', args.eps, args.k_value), allow_pickle=True)
    baseline_rank_result = np.load('./result/{}/{}/{}/eps_{}/k_{}.npy'.format(args.dataset, args.label_difficult, 'baseline_kfool', args.eps, args.k_value), allow_pickle=True)
    ta_norm_list = target_attack_result[0]
    ta_index = target_attack_result[1]
    base_index = baseline_rank_result[0]
    ta_dic = {}
    sample_list = np.load('ap_{}_list.npy'.format(args.dataset))

    for i in range(len(ta_index)):
        ta_dic[ta_index[i]]=ta_norm_list[i]

    ###find diff index and large norm
    # diff = []
    # key_list = []
    # index = 0
    # for ith, (data, GT) in tqdm(enumerate(valid_loader)):
    #     if ith in sample_list[:1000]:
    #         if np.sum(GT[0].numpy()) > 1:
    #             if ith in ta_index and ith not in base_index:
    #                 diff.append(ta_dic[ith])
    #                 key_list.append(ith)
    #         index = index + 1
    #     if index == 1000:
    #         break
    # top_1 = np.argsort(diff)[:10]
    # selected_index = np.asarray(key_list)[top_1]
    # print('selected_index:',selected_index)


    a = {}
    if args.app =='target_attack':
        a['best_3'] = [1118, 488, 460]
        a['best_5'] = [310, 316, 814]
        a['best_10'] = [858,896,316]
        a['random_3'] = [309, 860, 828]
        a['random_5'] = [721,390,603]
        a['random_10'] = [1067,978,896]
        a['worst_3'] = [858,859,721]
        a['worst_5'] = [942,870,978]
        a['worst_10'] = [1137,521,858]
    elif args.app =='none_target_attack':
        a['best_3'] = [5]  # 874, 49,5
        a['best_5'] = [5]
        a['best_10'] = [5]
    elif args.app =='UAP_attack' or args.app =='baseline_kUAP':
        a['best_3'] = [0, 6574, 7138, 8114, 8184] ##6472,6474,6475,6477,6479,6480,6481, 6482, 6483,6484
    # mean = [0.5, 0.5, 0.5]
    # std = [0.5, 0.5, 0.5]
    mean = 0.5
    std = 0.5
    index = 0
    if args.dataset=='VOC':
        labels=['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Cow', \
                'Diningtable', 'Dog', 'Horse', 'Motorbike', 'Person', 'Pottedplant', 'Sheep', 'Sofa', 'Train', 'Tvmonitor']
    else:
        labels=['person', 'bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',\
                'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',\
                'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',\
                'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',\
                'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange',\
                'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed',\
                'dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',\
                'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']
    for ith in a['{}_{}'.format(args.label_difficult, args.k_value)]:
        if ith ==0:
            pertub = np.load('./result/{}/{}/{}/eps_{}/perturbation_{}.npy'.format(args.dataset, args.label_difficult, args.app,
                                                                          args.eps, args.k_value))
            fig = plt.figure(constrained_layout=True)
            plt.imshow(pertub[0].transpose((1, 2, 0)))
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            os.makedirs('./plot_fig/{}/{}/{}/eps_{}'.format(args.dataset, args.label_difficult, args.app, args.eps),
                        exist_ok=True)
            fig.savefig('./plot_fig/{}/{}/{}/eps_{}/{}_UA_result_{}_k_{}_ith_{}.jpg'.format( \
                args.dataset, args.label_difficult, args.app, args.eps, args.dataset, args.label_difficult,
                args.k_value, ith),
                bbox_inches='tight')
            print('$||z||$={:.2f}'.format(np.linalg.norm(pertub[0])))
        else:
            index = index +1
            if index <= 100:
                ta_3 = np.load('./plot_result/{}/{}/{}/eps_{}/images_result_k_{}_ith_{}.npy'.format(args.dataset, args.label_difficult, args.app, args.eps,
                                                                          args.k_value,ith), allow_pickle=True)
                # bs_3 = np.load('./plot_result/{}/{}/{}/eps_{}/images_result_k_{}_ith_{}.npy'.format(args.dataset, args.label_difficult, 'baseline_kfool',
                #                                                                       args.eps, args.k_value, ith), allow_pickle=True)


                fig = plt.figure(constrained_layout=True)
                plt.imshow(ta_3[3][0].transpose((1, 2, 0)) * std + mean)
                plt.axis('off')
                plt.tight_layout()
                plt.show()

                GT = np.asarray(labels)[ta_3[1][0] == 1]
                GT_str = ''
                for i in range(GT.size):
                    GT_str += GT[i]
                    if i < GT.size-1:
                        GT_str += ','
                    if (i+1)%3==0 and (i+1)!=GT.size:
                        GT_str += '\n'
                TA = np.asarray(labels)[ta_3[4][0] == 1]
                TA_str = ''
                for j in range(TA.size):
                    TA_str += TA[j]
                    if j < TA.size - 1:
                        TA_str += ','
                    if (j + 1) % 3 == 0 and (j + 1) != TA.size:
                        TA_str += '\n'
                os.makedirs('./plot_fig/{}/{}/{}/eps_{}'.format(args.dataset,args.label_difficult, args.app, args.eps),
                            exist_ok=True)
                fig.savefig('./plot_fig/{}/{}/{}/eps_{}/{}_UA_result_{}_k_{}_ith_{}.jpg'.format(\
                    args.dataset,args.label_difficult, args.app, args.eps, args.dataset, args.label_difficult, args.k_value, ith),
                            bbox_inches='tight')
                print('GT:{}\nTop-3:{}'.format(GT_str, TA_str))
                print('shu')
            # if index ==10:
            #     break








# Execute main function here.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TKMLAP')
    parser.add_argument('--data', default='./data/COCO_2014', help='path to dataset')  ## ./data or ./data/COCO_2014
    parser.add_argument('--dataset', default='COCO', type=str, choices={'VOC', 'COCO'}, help='path to dataset')
    parser.add_argument('--results', default='woSigmoid-BCE-Adam-bs128-box_-1_1_COCO_fixMem',
                        help='path to dataset')  ##woSigmoid-BCE-Adam-bs64-box_-1_1, woSigmoid-BCE-Adam-bs128-box_-1_1_COCO_fixMem
    parser.add_argument('--num_classes', default=80, type=int, help='number of classes')  ##20 or 80
    parser.add_argument('--arch', default='resnet50',
                        help='model architecture: ' +
                             ' (default: inception_v3, resnet50)')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--image_size', default=224, type=int,
                        metavar='N', help='input image size (default: 300, 224)')
    parser.add_argument('--lr', '--learning-rate', default=[1.5e-4, 5e-2], type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--normalize', default='boxmaxmin', type=str, choices={'mean_std', 'boxmaxmin'},
                        help='optimizer for training')
    parser.add_argument('--opt', default='Adam', type=str, choices={'Adam', 'SGD'}, help='optimizer for training')
    parser.add_argument('--num', default=1, type=int, help='num to resume')
    parser.add_argument('--download_data', default=False, type=bool, help='download data')
    parser.add_argument('--save_results', default=True, type=bool, help='save results')
    parser.add_argument('--k_value', default=3, type=int, help='k-value')
    parser.add_argument('--eps', default=10, type=int, help='eps')
    parser.add_argument('--maxiter', default=1000, type=int, help='max iteration to attack')
    parser.add_argument('--remove_tier_para', default=0, type=float, help='remove_tier_para')
    parser.add_argument('--boxmax', default=1, type=float, help='max value of input')
    parser.add_argument('--boxmin', default=-1, type=float, help='min value of input')
    parser.add_argument('--lr_attack', default=1e-2, type=float, help='learning rate of attacks')
    parser.add_argument('--ufr_lower_bound', default=0.7, type=float, help='ufr lower bound')
    parser.add_argument('--max_iter_uni', default=np.inf, type=int, help='max iter in universal')
    parser.add_argument('--uap_train_index_end', default=3000, type=int, help='tain index in universal')
    parser.add_argument('--uap_test_index_start', default=3000, type=int, help='test index start in universal')
    parser.add_argument('--uap_test_index_end', default=4000, type=int, help='test index end in universal')
    parser.add_argument('--uap_norm', default=2, help='2 or np.inf')
    parser.add_argument('--uap_eps', default=100, type=int, help='eps for uap. 2000 for l_2 norm, 10 for l_infty norm.')
    parser.add_argument('--label_difficult', default='best', type=str, choices={'best', 'random', 'worst'},
                        help='difficult types')
    parser.add_argument('--app', default='baseline_kUAP', type=str, \
                        choices={'target_attack', 'none_target_attack', 'UAP_attack', 'baseline_rank', 'baseline_kfool',
                                 'baseline_kUAP', 'test', 'train'}, \
                        help='attack types')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu)
    model, device, valid_loader = main(args)
    plot(args,model, device, valid_loader)
