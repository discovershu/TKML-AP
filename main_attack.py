
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
from randaugment import RandAugment
import torch.optim as optim
from train import train_model, test
from attack import tkmlap
from baseline_attacks import baselineap
from utils import encode_labels, plot_history
import os
import torch.utils.model_zoo as model_zoo
import utils
from models.inception import Inception3

os.environ['TORCH_HOME'] = '.'
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def main(args):
    """
    Main function

    Args:
        data_dir: directory to download Pascal VOC data
        model_name: resnet18, resnet34 or resnet50
        num: model_num for file management purposes (can be any postive integer. Your results stored will have this number as suffix)
        lr: initial learning rate list [lr for resnet_backbone, lr for resnet_fc]
        epochs: number of training epochs
        batch_size: batch size. Default=16
        download_data: Boolean. If true will download the entire 2012 pascal VOC data as tar to the specified data_dir.
        Set this to True only the first time you run it, and then set to False. Default False
        save_results: Store results (boolean). Default False

    Returns:
        test-time loss and average precision

    Example way of running this function:
        if __name__ == '__main__':
            main('../data/', "resnet34", num=1, lr = [1.5e-4, 5e-2], epochs = 15, batch_size=16, download_data=False, save_results=True)
    """

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

    if args.dataset =='COCO':
        # COCO DataLoader
        instances_path_val = os.path.join(args.data, 'annotations/instances_val2014.json')
        instances_path_train = os.path.join(args.data, 'annotations/instances_train2014.json')
        # data_path_val = args.data
        # data_path_train = args.data
        data_path_val   = f'{args.data}/val2014'    # args.data
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
                                              transforms.RandomGrayscale(p = 0.25)
                                              ]),
                                          transforms.RandomHorizontalFlip(p = 0.25),
                                          transforms.RandomRotation(25),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean = mean, std = std),
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
                                              transforms.Normalize(mean = mean, std = std),
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
                                              transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                              transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean = mean, std = std)(crop) for crop in crops])),
                                              ])

        dataset_test = PascalVOC_Dataset(data_dir,
                                          year='2012',
                                          image_set='val',
                                          download=download_data,
                                          transform=transformations_test,
                                          target_transform=encode_labels)



    #---------------Test your model here---------------------------------------
    # Load the best weights before testing

    if args.app == 'train':
        log_file = open(os.path.join(model_dir, "log-{}.txt".format(num)), "w+")
        log_file.write("----------Experiment {} - {}-----------\n".format(num, model_name))
        # log_file.write("transformations == {}\n".format(transformations.__str__()))
        trn_hist, val_hist = train_model(model, device, optimizer, scheduler, train_loader, valid_loader, model_dir, num, epochs, log_file)
        torch.cuda.empty_cache()

        plot_history(trn_hist[0], val_hist[0], "Loss", os.path.join(model_dir, "loss-{}".format(num)))
        plot_history(trn_hist[1], val_hist[1], "Accuracy", os.path.join(model_dir, "accuracy-{}".format(num)))
        log_file.close()

    elif args.app == 'test':
        weights_file_path = os.path.join(model_dir, "model-{}.pth".format(num))
        if os.path.isfile(weights_file_path):
            print("Loading best weights")
            model.load_state_dict(torch.load(weights_file_path))
        ### we use val dataset without 5 crops for testing
        # test_loader = DataLoader(valid_loader, batch_size=int(batch_size), num_workers=0, shuffle=False)
        if args.save_results:
            loss, ap, scores, gt = test(model, device, valid_loader, returnAllScores=True, num_classes = args.num_classes)

            gt_path, scores_path, scores_with_gt_path = os.path.join(model_dir, "gt-{}.csv".format(num)), os.path.join(model_dir, "scores-{}.csv".format(num)), os.path.join(model_dir, "scores_wth_gt-{}.csv".format(num))

            utils.save_results(valid_loader.dataset.images, gt, utils.object_categories, gt_path)
            utils.save_results(valid_loader.dataset.images, scores, utils.object_categories, scores_path)
            utils.append_gt(gt_path, scores_path, scores_with_gt_path)

            utils.get_classification_accuracy(gt_path, scores_path, os.path.join(model_dir, "clf_vs_threshold-{}.png".format(num)))

            ap_1_list = []
            for i in range(len(gt)):
                score = utils.average_precision_score(gt[i], scores[i])
                if score ==1:
                    ap_1_list.append(i)
            np.save('ap_{}_list'.format(args.dataset), ap_1_list)

            print('Testing AP: {}'.format(scores/len(gt)))

    elif 'attack' in args.app:
        weights_file_path = os.path.join(model_dir, "model-{}.pth".format(num))
        if os.path.isfile(weights_file_path):
            print("Loading best weights")
            model.load_state_dict(torch.load(weights_file_path))
        tkmlap(args, model, device, valid_loader)

    elif 'baseline' in args.app:
        weights_file_path = os.path.join(model_dir, "model-{}.pth".format(num))
        if os.path.isfile(weights_file_path):
            print("Loading best weights")
            model.load_state_dict(torch.load(weights_file_path))
        baselineap(args, model, device, valid_loader)



# Execute main function here.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TKMLAP')
    parser.add_argument('--data', default='./data', help='path to dataset') ## ./data or ./data/COCO_2014
    parser.add_argument('--dataset', default='VOC', type=str, choices={'VOC', 'COCO'}, help='path to dataset')
    parser.add_argument('--results', default='woSigmoid-BCE-Adam-bs64-box_-1_1', help='path to dataset') ##woSigmoid-BCE-Adam-bs64-box_-1_1, woSigmoid-BCE-Adam-bs128-box_-1_1_COCO_fixMem
    parser.add_argument('--num_classes', default=20, type=int, help='number of classes') ##20 or 80
    parser.add_argument('--arch',  default='inception_v3',
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
    parser.add_argument('--image_size', default=300, type=int,
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
    parser.add_argument('--ufr_lower_bound', default=0.8, type=float, help='ufr lower bound')
    parser.add_argument('--max_iter_uni', default=np.inf, type=int, help='max iter in universal')
    parser.add_argument('--uap_train_index_end', default=3000, type=int, help='tain index in universal')
    parser.add_argument('--uap_test_index_start', default=3000, type=int, help='test index start in universal')
    parser.add_argument('--uap_test_index_end', default=4000, type=int, help='test index end in universal')
    parser.add_argument('--uap_norm', default=2, help='2 or np.inf')
    parser.add_argument('--uap_eps', default=100, type=int, help='eps for uap. 2000 for l_2 norm, 10 for l_infty norm.')
    parser.add_argument('--label_difficult', default='best', type=str, choices={'best', 'random', 'worst'}, help='difficult types')
    parser.add_argument('--app', default='none_target_attack', type=str, \
                        choices={'target_attack', 'none_target_attack', 'UAP_attack', 'baseline_rank', 'baseline_kfool', 'baseline_kUAP', 'test', 'train'}, \
                        help='attack types')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu)
    # main('./data/', "inception_v3", num=1, lr = [1.5e-4, 5e-2], epochs = 100, batch_size=1, download_data=False, save_results=True)
    main(args)
