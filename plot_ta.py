
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

os.environ['TORCH_HOME'] = '.'
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def main(args):
    data_dir = args.data
    model_name = args.arch
    num = args.num
    batch_size = args.batch_size
    download_data = args.download_data

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
    device = torch.device("cuda" if use_cuda else "cpu")

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

    # VOC validation dataloader
    transformations_valid = transforms.Compose([transforms.Resize(330),
                                          transforms.CenterCrop(300),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean = mean, std = std),
                                          ])

    # transformations_valid = transforms.Compose([transforms.ToTensor(),
    #                                             ])

    VOC_dataset_valid = PascalVOC_Dataset(data_dir,
                                      year='2012',
                                      image_set='val',
                                      download=download_data,
                                      transform=transformations_valid,
                                      target_transform=encode_labels)
    valid_loader = DataLoader(VOC_dataset_valid, batch_size=batch_size, num_workers=4)


    #---------------Test your model here---------------------------------------
    # Load the best weights before testing
    weights_file_path = os.path.join(model_dir, "model-{}.pth".format(num))
    if os.path.isfile(weights_file_path):
        print("Loading best weights")
        model.load_state_dict(torch.load(weights_file_path))

    # for ith, (data, GT) in tqdm(enumerate(valid_loader)):
    #     if ith in [521]:
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
    model_name = args.arch
    model_dir = os.path.join(args.results, args.arch)
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
    device = torch.device("cuda" if use_cuda else "cpu")

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

    model.eval()
    target_attack_result = np.load('./result/{}/{}/{}/eps_{}/k_{}.npy'.format(args.dataset, args.label_difficult, 'target_attack', args.eps, args.k_value), allow_pickle=True)
    baseline_rank_result = np.load('./result/{}/{}/{}/eps_{}/k_{}.npy'.format(args.dataset, args.label_difficult, 'baseline_rank', args.eps, args.k_value), allow_pickle=True)
    ta_norm_list = target_attack_result[0]
    # base_norm_list =baseline_rank_result[0]
    ta_index = target_attack_result[1]
    base_index = baseline_rank_result[0]
    ta_dic = {}
    base_dic = {}
    sample_list = np.load('ap_{}_list.npy'.format(args.dataset))

    for i in range(len(ta_index)):
        ta_dic[ta_index[i]]=ta_norm_list[i]
    # for j in range(len(base_index)):
    #     base_dic[base_index[j]]=base_norm_list[j]
    ###find same index and small norm
    # index = 0
    # same = []
    # key_list = []
    # for ith, (data, GT) in tqdm(enumerate(valid_loader)):
    #     if ith in sample_list[:1000]:
    #         if np.sum(GT[0].numpy())>1:
    #             if ith in ta_index and ith in base_index:
    #                 same.append(np.abs(ta_dic[ith] -base_dic[ith]))
    #                 key_list.append(ith)
    #         index = index + 1
    #     if index ==1000:
    #         break
    # top_3 = np.argsort(same)[:3]
    # selected_index = np.asarray(key_list)[top_3]
    # print('selected_index:',selected_index)

    ###find diff index and large norm
    # diff = []
    # key_list = []
    # index = 0
    # for ith, (data, GT) in tqdm(enumerate(valid_loader)):
    #     if ith in sample_list[:1000]:
    #         if np.sum(GT[0].numpy()) > 1:
    #             if ith in ta_index and ith in base_index:
    #             # if ith in ta_index:
    #                 diff.append(ta_dic[ith])
    #                 key_list.append(ith)
    #         index = index + 1
    #     if index == 1000:
    #         break
    # top_3 = np.argsort(diff)[:1000]
    # selected_index = np.asarray(key_list)[np.sort(top_3)]
    # print('selected_index:',selected_index)


    a = {}
    if args.eps == 10:
        a['best_3'] = [1118, 488, 460]
        a['best_5'] = [310, 316, 814]
        a['best_10'] = [858,896,316]
        a['random_3'] = [309, 860, 828]
        a['random_5'] = [721,390,603]
        a['random_10'] = [1067,978,896]
        a['worst_3'] = [858,859,721]
        a['worst_5'] = [942,870,978]
        a['worst_10'] = [1137,521,858]
    elif args.eps == 2:
        a['best_3'] = [277] #[277,313,412]
        a['best_5'] = [277] #[277,313,412]
        a['best_10'] = [277] #[277,313,412]
        a['random_3'] = [247] #[247,9,  194]
        # a['random_3'] = [521]  # [247,9,  194]
        a['random_5'] = [247] #[247, 874 ,746]
        a['random_10'] = [247] #[247, 386, 720]
        a['worst_3'] = [647] #[614, 647, 1177]
        a['worst_5'] = [647] #[614, 647, 1177]
        a['worst_10'] = [647] #[614, 647, 1177]
    # mean = [0.5, 0.5, 0.5]
    # std = [0.5, 0.5, 0.5]
    mean = 0.5
    std = 0.5
    index = 0
    labels=['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Cow', \
            'Diningtable', 'Dog', 'Horse', 'Motorbike', 'Person', 'Pottedplant', 'Sheep', 'Sofa', 'Train', 'Tvmonitor']
    for ith in a['{}_{}'.format(args.label_difficult, args.k_value)]:
        index = index +1
        if index <= 1:
            ta = np.load('./plot_result/{}/{}/{}/eps_{}/images_result_k_{}_ith_{}.npy'.format(args.dataset, args.label_difficult, 'target_attack', args.eps,
                                                                      args.k_value, ith), allow_pickle=True)
            bs = np.load('./plot_result/{}/{}/{}/eps_{}/images_result_k_{}_ith_{}.npy'.format(args.dataset, args.label_difficult, 'baseline_rank',
                                                                                  args.eps, args.k_value, ith), allow_pickle=True)

            ta_word_3 =np.load('./plot_result/{}/{}/{}/eps_{}/images_result_k_3_ith_{}.npy'.format(args.dataset, args.label_difficult, 'target_attack', args.eps,
                                                                      ith), allow_pickle=True)
            ta_word_5 = np.load(
                './plot_result/{}/{}/{}/eps_{}/images_result_k_5_ith_{}.npy'.format(args.dataset, args.label_difficult,
                                                                                    'target_attack', args.eps,
                                                                                    ith), allow_pickle=True)
            ta_word_10 = np.load(
                './plot_result/{}/{}/{}/eps_{}/images_result_k_10_ith_{}.npy'.format(args.dataset, args.label_difficult,
                                                                                    'target_attack', args.eps,
                                                                                    ith), allow_pickle=True)

            os.makedirs('./plot_fig/{}/{}/{}/eps_{}'.format(args.dataset, args.label_difficult, args.app, args.eps),
                        exist_ok=True)
            ########GT
            fig = plt.figure(constrained_layout=True)
            plt.imshow(ta[0][0].transpose((1, 2, 0)) * std + mean)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            if args.k_value == 3:
                fig.savefig('./plot_fig/{}/{}/{}/eps_{}/{}_TA_result_{}_k_{}_ith_{}_original.jpg'.format( \
                    args.dataset, args.label_difficult, args.app, args.eps, args.dataset, args.label_difficult,
                    args.k_value, ith),
                    bbox_inches='tight')
            GT = np.asarray(labels)[ta[1][0] == 1]
            GT_str = ''
            for i in range(GT.size):
                GT_str += GT[i]
                if i < GT.size - 1:
                    GT_str += ','
            TA = np.asarray(labels)[ta[3][0] == 1]
            TA_str_3 = ''
            # for m in range(3):
            for j in range(TA.size):
                TA_str_3 += TA[j]
                if j < TA.size - 1:
                    TA_str_3 += ','
            TA_5 = np.asarray(labels)[ta_word_5[3][0] == 1]
            TA_str_5 = ''
            # for m in range(3):
            for j in range(TA_5.size):
                TA_str_5 += TA_5[j]
                if j < TA_5.size - 1:
                    TA_str_5 += ','
            TA_10 = np.asarray(labels)[ta_word_10[3][0] == 1]
            TA_str_10 = ''
            # for m in range(3):
            for j in range(TA_10.size):
                TA_str_10 += TA_10[j]
                if j < TA_10.size - 1:
                    TA_str_10 += ','
            print('GT:{}\n\nTA-3:{}\n\nTA-5:{}\n\nTA-10:{}'.format(GT_str, TA_str_3, TA_str_5, TA_str_10),
                           )

            ########MLAP
            fig = plt.figure(constrained_layout=True)
            plt.imshow(bs[4][0].transpose((1, 2, 0)) * std + mean)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            fig.savefig('./plot_fig/{}/{}/{}/eps_{}/{}_TA_result_{}_k_{}_ith_{}_mlap.jpg'.format( \
                args.dataset, args.label_difficult, args.app, args.eps, args.dataset, args.label_difficult,
                args.k_value, ith),
                bbox_inches='tight')
            ML_AP_TA = np.asarray(labels)[bs[6][0] == 1]
            ML_AP_TA_str = ''
            for j in range(ML_AP_TA.size):
                ML_AP_TA_str += ML_AP_TA[j]
                if j < ML_AP_TA.size - 1:
                    ML_AP_TA_str += ','
            print('Top_mlap-{}:{}'.format(args.k_value, ML_AP_TA_str))

            fig = plt.figure(constrained_layout=True)
            plt.imshow(20 * (bs[7][0].transpose((1, 2, 0))))
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            fig.savefig('./plot_fig/{}/{}/{}/eps_{}/{}_TA_result_{}_k_{}_ith_{}_mlap_pert.jpg'.format( \
                args.dataset, args.label_difficult, args.app, args.eps, args.dataset, args.label_difficult,
                args.k_value, ith),
                bbox_inches='tight')
            print('$||z||$={:.2f}'.format(np.linalg.norm(bs[7][0])))

            ########TKML-TA
            fig = plt.figure(constrained_layout=True)
            plt.imshow(ta[4][0].transpose((1, 2, 0)) * std + mean)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            fig.savefig('./plot_fig/{}/{}/{}/eps_{}/{}_TA_result_{}_k_{}_ith_{}_tkml-ta.jpg'.format( \
                args.dataset, args.label_difficult, args.app, args.eps, args.dataset, args.label_difficult,
                args.k_value, ith),
                bbox_inches='tight')
            TKML_TA = np.asarray(labels)[ta[6][0] == 1]
            TKML_TA_str = ''
            for j in range(TKML_TA.size):
                TKML_TA_str += TKML_TA[j]
                if j < TKML_TA.size - 1:
                    TKML_TA_str += ','
            print('Top_tkml-ta-{}:{}'.format(args.k_value, TKML_TA_str))

            fig = plt.figure(constrained_layout=True)
            plt.imshow(20 * (ta[7][0].transpose((1, 2, 0))))
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            fig.savefig('./plot_fig/{}/{}/{}/eps_{}/{}_TA_result_{}_k_{}_ith_{}_tkml-ta_pert.jpg'.format( \
                args.dataset, args.label_difficult, args.app, args.eps, args.dataset, args.label_difficult,
                args.k_value, ith),
                bbox_inches='tight')
            print('$||z||$={:.2f}'.format(np.linalg.norm(ta[7][0])))

            # fig = plt.figure(constrained_layout=True)
            # gs = gridspec.GridSpec(4, 3)
            # gs.update(wspace=0.5)
            # ax1 = fig.add_subplot(gs[0:2, 0])
            # ax2 = fig.add_subplot(gs[:2, 1])
            # ax3 = fig.add_subplot(gs[:2, 2])
            # ax4 = fig.add_subplot(gs[2:, 1])
            # ax5 = fig.add_subplot(gs[2:, 2])
            # # ax1 = plt.subplot(gs[1:3, 0])
            # # ax2 = plt.subplot(gs[:2, 1])
            # # ax3 = plt.subplot(gs[:2, 2])
            # # ax4 = plt.subplot(gs[2:, 1])
            # # ax5 = plt.subplot(gs[2:, 2])
            # ax1.imshow(ta[0][0].transpose((1, 2, 0)) * std + mean)
            # ax1.axes.xaxis.set_ticks([])
            # ax1.axes.yaxis.set_ticks([])
            # ax1.set_title('(1) Originial Image', fontsize=8)
            # GT = np.asarray(labels)[ta[1][0] == 1]
            # GT_str = ''
            # for i in range(GT.size):
            #     GT_str += GT[i]
            #     if i < GT.size-1:
            #         GT_str += ','
            #     if (i+1)%3==0 and (i+1)!=GT.size:
            #         GT_str += '\n'
            # TA = np.asarray(labels)[ta[3][0] == 1]
            # TA_str_3 = ''
            # # for m in range(3):
            # for j in range(TA.size):
            #     TA_str_3 += TA[j]
            #     if j < TA.size-1:
            #         TA_str_3 += ','
            #     if (j+1)%2==0and (j+1)!=TA.size:
            #         TA_str_3 += '\n'
            # TA_5 = np.asarray(labels)[ta_word_5[3][0] == 1]
            # TA_str_5 = ''
            # # for m in range(3):
            # for j in range(TA_5.size):
            #     TA_str_5 += TA_5[j]
            #     if j < TA_5.size - 1:
            #         TA_str_5 += ','
            #     if (j + 1) % 2 == 0 and (j + 1) != TA_5.size:
            #         TA_str_5 += '\n'
            # TA_10 = np.asarray(labels)[ta_word_10[3][0] == 1]
            # TA_str_10 = ''
            # # for m in range(3):
            # for j in range(TA_10.size):
            #     TA_str_10 += TA_10[j]
            #     if j < TA_10.size - 1:
            #         TA_str_10 += ','
            #     if (j + 1) % 2 == 0 and (j + 1) != TA_10.size:
            #         TA_str_10 += '\n'
            # ax1.set_xlabel('GT:{}\n\nTA-3:{}\n\nTA-5:{}\n\nTA-10:{}'.format(GT_str,TA_str_3,TA_str_5,TA_str_10), fontsize=8, wrap=True)
            #
            # ax2.imshow(bs[4][0].transpose((1, 2, 0)) * std + mean)
            # ML_AP_TA = np.asarray(labels)[bs[6][0] == 1]
            # ML_AP_TA_str = ''
            # for j in range(ML_AP_TA.size):
            #     ML_AP_TA_str += ML_AP_TA[j]
            #     if j < ML_AP_TA.size - 1:
            #         ML_AP_TA_str += ','
            #     if (j+1)%3==0and (j+1)!=ML_AP_TA.size:
            #         ML_AP_TA_str += '\n'
            # ax2.set_xlabel('Top-{}:{}'.format(args.k_value, ML_AP_TA_str), fontsize=8)
            # ax2.axes.xaxis.set_ticks([])
            # ax2.axes.yaxis.set_ticks([])
            # if args.k_value ==3:
            #     ax2.set_title('(2) ML-AP Advers. Image', fontsize=8)
            # elif args.k_value ==5:
            #     ax2.set_title('(6) ML-AP Advers. Image', fontsize=8)
            # else:
            #     ax2.set_title('(10) ML-AP Advers. Image', fontsize=8)
            # temp_0 = bs[7][0].transpose((1, 2, 0)) * std
            # # temp_0 = (temp_0 - temp_0.min()) / (temp_0.max() - temp_0.min())
            # ax3.imshow(0*(ta[0][0].transpose((1, 2, 0)) * std + mean)+50*temp_0)
            # ax3.axes.xaxis.set_ticks([])
            # ax3.axes.yaxis.set_ticks([])
            # ax3.set_xlabel('$||z||$={:.2f}'.format(np.linalg.norm(bs[7][0])), fontsize=8)
            # if args.k_value == 3:
            #     ax3.set_title('(3) ML-AP Perturbation', fontsize=8)
            # elif args.k_value == 5:
            #     ax3.set_title('(7) ML-AP Perturbation', fontsize=8)
            # else:
            #     ax3.set_title('(11) ML-AP Perturbation', fontsize=8)
            #
            # ax4.imshow(ta[4][0].transpose((1, 2, 0)) * std + mean)
            # TKML_TA = np.asarray(labels)[ta[6][0] == 1]
            # TKML_TA_str = ''
            # for j in range(TKML_TA.size):
            #     TKML_TA_str += TKML_TA[j]
            #     if j < TKML_TA.size - 1:
            #         TKML_TA_str += ','
            #     if (j+1)%3==0and (j+1)!=TKML_TA.size:
            #         TKML_TA_str += '\n'
            # ax4.set_xlabel('Top-{}:{}'.format(args.k_value, TKML_TA_str), fontsize=8)
            # ax4.axes.xaxis.set_ticks([])
            # ax4.axes.yaxis.set_ticks([])
            # if args.k_value == 3:
            #     ax4.set_title('(4) TKML-TA Advers. Image', fontsize=8)
            # elif args.k_value == 5:
            #     ax4.set_title('(8) TKML-TA Advers. Image', fontsize=8)
            # else:
            #     ax4.set_title('(12) TKML-TA Advers. Image', fontsize=8)
            #
            # temp = ta[7][0].transpose((1, 2, 0)) * std
            # # temp = (temp-temp.min())/(temp.max()-temp.min())
            # ax5.imshow(0*(ta[0][0].transpose((1, 2, 0)) * std + mean)+50*temp)
            # ax5.set_xlabel('$||z||$={:.2f}'.format(np.linalg.norm(ta[7][0])), fontsize=8)
            # ax5.axes.xaxis.set_ticks([])
            # ax5.axes.yaxis.set_ticks([])
            # if args.k_value == 3:
            #     ax5.set_title('(5) TKML-TA Perturbation', fontsize=8)
            # elif args.k_value == 5:
            #     ax5.set_title('(9) TKML-TA Perturbation', fontsize=8)
            # else:
            #     ax5.set_title('(13) TKML-TA Perturbation', fontsize=8)
            #
            # plt.tight_layout()
            # plt.show()
            # os.makedirs('./plot_fig/{}/{}/{}/eps_{}'.format(args.dataset,args.label_difficult, args.app, args.eps),
            #             exist_ok=True)
            # fig.savefig('./plot_fig/{}/{}/{}/eps_{}/{}_TA_result_{}_k_{}_ith_{}.jpg'.format(\
            #     args.dataset,args.label_difficult, args.app, args.eps, args.dataset, args.label_difficult, args.k_value, ith),
            #             bbox_inches='tight')
            # print('shu')
        if index ==1:
            break








# Execute main function here.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TKMLAP')
    parser.add_argument('--data', default='./data', help='path to dataset')
    parser.add_argument('--dataset', default='VOC', type=str, choices={'VOC', 'COCO'}, help='path to dataset')
    parser.add_argument('--results', default='woSigmoid-BCE-Adam-bs64-box_-1_1', help='path to dataset')
    parser.add_argument('--num_classes', default=20, type=int, help='number of classes')
    parser.add_argument('--arch',  default='inception_v3',
                        help='model architecture: ' +
                            ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--normalize', default='boxmaxmin', type=str, choices={'mean_std', 'boxmaxmin'},
                        help='optimizer for training')
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--download_data', default=False, type=bool, help='download data')
    parser.add_argument('--num', default=1, type=int, help='num to resume')
    parser.add_argument('--k_value', default=10, type=int, help='k-value')
    parser.add_argument('--eps', default=2, type=int, help='eps')
    parser.add_argument('--boxmax', default=1, type=float, help='max value of input')
    parser.add_argument('--boxmin', default=-1, type=float, help='min value of input')
    parser.add_argument('--label_difficult', default='worst', type=str, choices={'best', 'random', 'worst'},
                        help='difficult types')
    parser.add_argument('--app', default='target_attack', type=str,
                        choices={'target_attack', 'none_target_attack', 'UAP_attack', 'baseline_rank', 'baseline_kfool',
                                 'baseline_kUAP', 'test', 'train'}, help='attack types')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu)
    model, device, valid_loader = main(args)
    plot(args,model, device, valid_loader)

    # elif args.eps == 2:
    #     a['best_3'] = [961, 705,1045]
    #     a['best_5'] = [961,390,1045]
    #     a['best_10'] = [961, 313, 647]
    #     a['random_3'] = [247,9,  194]
    #     a['random_5'] = [247, 874 ,746]
    #     a['random_10'] = [247, 386, 720]