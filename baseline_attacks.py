from tqdm import tqdm
import torch
import gc
import os
from utils import get_ap_score, generate_target_zeros, generate_target_zeros_3_cases
import numpy as np
import torch
import numpy as np
from evaluate_metrics import hamming_loss, FR, TP_index, nontargeted_TP_index, topk_acc_metric, topk_acc_metric_1_to_10
from models.projection import clip_eta
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import time
import copy
import cv2

# def rank(model, inputs, label, k_value, maxiter, boxmax, boxmin, GT, binary_search_steps, gamma, device, remove_tier_para=0.1,lr=1e-2):
#     # trick from CW, normalize to [boxin, boxmin]
#     boxmul = (boxmax - boxmin) / 2.0
#     boxplus = (boxmax + boxmin) / 2.0
#
#     timg = inputs
#     tlab = label
#     const = gamma
#     shape = inputs.shape
#
#     # the best norm, score, and image attack
#     o_best_norm = 1e10
#     upper_bound = 1e10
#     lower_bound = 0
#     purtabed_img_out = torch.zeros(*shape)
#     modifier_out = torch.zeros(*shape)
#     attack_success = False
#     inner_attack_success = False
#     c_list = [0] * 6
#     best_outer_step = -1
#
#     for outer_step in range(binary_search_steps):
#         best_norm = 1e10
#
#         # variables we are going to optimize over
#         modifier = Variable(torch.zeros(*shape).cuda(), requires_grad=True).to(device)
#         model.eval()
#
#         optimizer = torch.optim.SGD([{'params': modifier}], lr=lr)
#
#         for iteration in range(maxiter):
#             optimizer.zero_grad()
#
#             purtabed_img = torch.tanh(modifier + timg) * boxmul + boxplus
#             purtabed_out = model(purtabed_img)
#             eta = purtabed_img-(torch.tanh(timg) * boxmul + boxplus)
#
#             l2dist = torch.sum(torch.square(eta), [1])
#             loss0 = torch.sum(l2dist)
#
#             real = torch.exp(torch.max((1-tlab) * purtabed_out - tlab*10000))
#             other = torch.exp(torch.min(tlab* purtabed_out+(1-tlab)*10000))
#             loss1 = torch.sum(torch.clamp(real-other, min=0.0))
#
#             loss = loss0 + const*loss1
#
#             # Calculate gradient
#             loss.backward()
#             optimizer.step()
#
#             TP_flag, predict_label = TP_index(tlab.cpu().detach().numpy(), purtabed_out.cpu().detach().numpy())
#             eta_norm = np.linalg.norm(eta.cpu().detach().numpy())
#             if TP_flag==True and eta_norm<best_norm:
#                 best_norm = eta_norm
#                 inner_attack_success = True
#             if TP_flag==True and eta_norm<o_best_norm:
#                 o_best_norm = eta_norm
#                 purtabed_img_out = np.arctanh(((purtabed_img - boxplus) / boxmul* 0.999999).cpu().detach().numpy())
#                 modifier_out = purtabed_img_out - timg.cpu().detach().numpy()
#                 attack_success = True
#
#                 c_list[0], c_list[1], c_list[2], c_list[3], c_list[4], c_list[5] = \
#                     topk_acc_metric_1_to_10(GT.cpu().detach().numpy(), purtabed_out.cpu().detach().numpy())
#                 best_outer_step = outer_step
#
#             print('outer_iter:', outer_step, 'inner_iter:', iteration, \
#                   'best_outer_step= ', "{}".format(best_outer_step), \
#                   'loss= ', "{:.5f}".format(loss), \
#                   'lambda= ', "{:.2f}".format(const), \
#                   'loss0= ', "{:.5f}".format(loss0), \
#                   'loss1= ', "{:.5f}".format(loss1), \
#                   'attacked: ', TP_flag, \
#                   'inner_attacked: ', inner_attack_success, \
#                   'targets: ', label.cpu().detach().numpy(), \
#                   'predict_label:', predict_label, \
#                   'GT:', GT.cpu().detach().numpy(), \
#                   'min:', "{:.5f}".format(modifier.min().cpu()), \
#                   'max:', "{:.5f}".format(modifier.max().cpu()), \
#                   'norm:', "{:.5f}".format(eta_norm))
#
#         if inner_attack_success == True:
#             upper_bound = min(upper_bound, const)
#             if upper_bound < 1e9:
#                 const = (lower_bound + upper_bound) / 2
#         else:
#             lower_bound = max(lower_bound, const)
#             if upper_bound < 1e9:
#                 const = (lower_bound + upper_bound) / 2
#             else:
#                 const *= 10
#         inner_attack_success = False
#
#     return purtabed_img_out, modifier_out, attack_success, c_list

def rank2(model, inputs, label, k_value, eps, maxiter, boxmax, boxmin, GT, device, lr=1e-2):
    # trick from CW, normalize to [boxin, boxmin]
    boxmul = (boxmax - boxmin) / 2.0
    boxplus = (boxmax + boxmin) / 2.0

    timg = inputs
    tlab = label
    const = eps
    shape = inputs.shape

    # variables we are going to optimize over
    if device.type == 'cuda':
        modifier = Variable(torch.zeros(*shape).cuda(), requires_grad=True).to(device)
    else:
        modifier = Variable(torch.zeros(*shape), requires_grad=True).to(device)
    model.eval()

    optimizer = torch.optim.SGD([{'params': modifier}], lr=lr)

    best_norm = 1e10
    purtabed_img_out = torch.zeros(*shape)
    modifier_out = torch.zeros(*shape)
    attack_success = False
    c_list = [0] * 6

    for iteration in range(maxiter):
        optimizer.zero_grad()

        purtabed_img = torch.tanh(modifier + timg) * boxmul + boxplus
        eta = purtabed_img - (torch.tanh(timg) * boxmul + boxplus)
        eta = clip_eta(eta, 2, const)

        purtabed_img = torch.tanh(timg) * boxmul + boxplus + eta
        purtabed_out = model(purtabed_img)

        # l2dist = torch.sum(torch.square(eta), [1])
        # loss0 = torch.sum(l2dist)

        real = torch.max((1 - tlab) * purtabed_out - tlab * 10000)
        other = torch.min(tlab * purtabed_out + (1 - tlab) * 10000)
        loss = torch.sum(torch.clamp(real - other, min=0.0))

        # Calculate gradient
        loss.backward()
        optimizer.step()

        TP_flag, predict_label = TP_index(tlab.cpu().detach().numpy(), purtabed_out.cpu().detach().numpy())
        eta_norm = np.linalg.norm(eta.cpu().detach().numpy())
        if TP_flag==True and eta_norm<best_norm:
            best_norm = eta_norm
            purtabed_img_out = np.arctanh(((purtabed_img - boxplus) / boxmul* 0.999999).cpu().detach().numpy())
            modifier_out = purtabed_img_out - timg.cpu().detach().numpy()
            attack_success = True

            c_list[0],c_list[1],c_list[2],c_list[3],c_list[4],c_list[5] = \
                topk_acc_metric_1_to_10(GT.cpu().detach().numpy(), purtabed_out.cpu().detach().numpy())

            print('iter:', iteration, 'loss= ', "{:.5f}".format(loss), \
                'attacked: ', TP_flag, \
                'targets: ', label.cpu().detach().numpy(),\
                'predict_label:', predict_label, \
                'GT:', GT.cpu().detach().numpy(), \
                'min:', "{:.5f}".format(modifier.min().cpu()), \
                'max:', "{:.5f}".format(modifier.max().cpu()), \
                'norm:', "{:.5f}".format(eta_norm) )

    return purtabed_img_out, modifier_out, attack_success, c_list

def jacobian(predictions, x, nb_classes):
    list_derivatives = []

    for class_ind in range(nb_classes):
        outputs = predictions[:, class_ind]
        derivatives, = torch.autograd.grad(outputs, x, grad_outputs=torch.ones_like(outputs), retain_graph=True)
        list_derivatives.append(derivatives)

    return list_derivatives

def kfool(model, inputs, GT, k_value, boxmax, boxmin,maxiter, device, lr=1e-2):
    boxmul = (boxmax - boxmin) / 2.0
    boxplus = (boxmax + boxmin) / 2.0

    timg = inputs
    tlab = GT
    shape = inputs.shape
    purtabed_img_out = np.zeros(shape)
    modifier_out = np.zeros(shape)

    all_list = set(list(range(0, tlab.cpu().detach().numpy().shape[1])))
    with torch.no_grad():
        F = model(torch.tanh(timg) * boxmul + boxplus)
    nb_classes = F.size(-1)

    purtabed_img = (torch.tanh(timg) * boxmul + boxplus).clone().requires_grad_()

    loop_i = 0
    attack_success = False
    c_list = [0] * 6
    F = model(purtabed_img)
    max_label = torch.argmax(tlab * F - (1 - tlab) * 10000)
    p = torch.argsort(F, dim=1, descending=True)
    tlab_all = ((tlab == 1).nonzero(as_tuple=True)[1]).cpu().detach().numpy()
    complement_set = all_list - set((p[0][:k_value]).cpu().detach().numpy())

    r_tot = torch.zeros(timg.size()).to(device)

    while (complement_set.issuperset(set(tlab_all))) == False and loop_i < maxiter:
        w = torch.squeeze(torch.zeros(timg.size()[1:])).to(device)
        f = 0
        top_F = F.topk(k_value + 1)[0]
        max_F = F[0][max_label].reshape((1,1))
        gradients_top = torch.stack(jacobian(top_F, purtabed_img, k_value + 1), dim=1)
        gradients_max = torch.stack(jacobian(max_F, purtabed_img, 1), dim=1)
        # gradients = torch.stack(jacobian(F, purtabed_img, nb_classes), dim=1)
        with torch.no_grad():
            for idx in range(inputs.size(0)):
                for k in range(k_value + 1):
                    if torch.all(torch.eq(gradients_top[idx, k, ...], gradients_max[idx,0,...]))==False and p[0][k]!=max_label:
                        norm = torch.div(1, torch.norm(gradients_top[idx, k, ...] - gradients_max[idx,0,...]))
                        w = w + (gradients_top[idx, k, ...] - gradients_max[idx,0,...]) * norm
                        f = f + (F[idx, p[0][k]] - F[idx, max_label]) * norm
                r_tot[idx, ...] = r_tot[idx, ...] + torch.abs(f) * w / torch.norm(w)
        purtabed_img = (torch.tanh(r_tot + timg) * boxmul + boxplus).requires_grad_()
        F = model(purtabed_img)
        p = torch.argsort(F, dim=1, descending=True)

        # complement_set = all_list - set(p[:k_value])
        complement_set = all_list - set((p[0][:k_value]).cpu().detach().numpy())
        # Flag, predict_label = nontargeted_TP_index(tlab.cpu().detach().numpy(), F.cpu().detach().numpy(), k_value)
        if complement_set.issuperset(set(tlab_all)):
            Flag, predict_label = nontargeted_TP_index(tlab.cpu().detach().numpy(), F.cpu().detach().numpy(), k_value)
            purtabed_img_out = np.arctanh(((purtabed_img - boxplus) / boxmul * 0.999999).cpu().detach().numpy())
            modifier_out = purtabed_img_out - timg.cpu().detach().numpy()
            attack_success = True
            c_list[0], c_list[1], c_list[2], c_list[3], c_list[4], c_list[5] = \
                topk_acc_metric_1_to_10(GT.cpu().detach().numpy(), F.cpu().detach().numpy())
            print('iter:', loop_i + 1, \
                  'attacked: ', attack_success, \
                  'predict_label:', predict_label, \
                  'GT:', tlab.cpu().detach().numpy(), \
                  'min:', "{:.5f}".format(r_tot.min().cpu()), \
                  'max:', "{:.5f}".format(r_tot.max().cpu()), \
                  'norm:', "{:.5f}".format(np.linalg.norm(r_tot.cpu().detach().numpy())))
        loop_i = loop_i + 1
    # boxmul = (boxmax - boxmin) / 2.0
    # boxplus = (boxmax + boxmin) / 2.0
    #
    # timg = inputs
    # tlab = GT
    # shape = inputs.shape
    #
    # all_list = set(list(range(0, tlab.cpu().detach().numpy().shape[1])))
    #
    #
    # purtabed_img = Variable((torch.tanh(timg) * boxmul + boxplus), requires_grad=True).to(device)
    # # purtabed_img = ((torch.tanh(timg) * boxmul + boxplus)).requires_grad_()
    # purtabed_img_out = np.zeros(shape)
    # modifier_out = np.zeros(shape)
    #
    # modifier = torch.zeros(*shape).cuda()
    #
    # F = model(purtabed_img)
    # p = torch.argsort(F, dim=1, descending=True)
    # tlab_all = ((tlab == 1).nonzero(as_tuple=True)[1]).cpu().detach().numpy()
    # complement_set = all_list - set((p[0][:k_value]).cpu().detach().numpy())
    # # p = F.cpu().detach().numpy().flatten().argsort()[::-1]
    # # tlab_all = np.transpose(np.argwhere(tlab.cpu().detach().numpy().flatten() == 1)).flatten()
    # # complement_set = all_list - set(p[:k_value])
    # loop_i = 0
    # attack_success = False
    # c_list = [0] * 6
    #
    # while (complement_set.issuperset(set(tlab_all)))==False and loop_i < maxiter:
    #     w = np.zeros(shape)
    #     # w = torch.zeros(shape).cuda()
    #     f = 0
    #     max_label = torch.argmax(tlab * F - (1 - tlab) * 10000)
    #     F[0, max_label].backward(retain_graph=True)
    #
    #     grad_ori = purtabed_img.grad.data.cpu().numpy().copy()
    #     # grad_ori = copy.deepcopy(purtabed_img.grad.data)
    #     # grad_ori = torch.autograd.grad(F[0, max_label], purtabed_img, grad_outputs=torch.ones_like(F[0, max_label]),
    #     #                     retain_graph=True)[0]
    #
    #     for i in range(1, k_value+1):
    #         zero_gradients(purtabed_img)
    #         F[0, p[0][i]].backward(retain_graph=True)
    #         cur_grad = purtabed_img.grad.data.cpu().numpy().copy()
    #         w = w + (cur_grad - grad_ori)/np.linalg.norm((cur_grad - grad_ori).flatten())
    #         f = f + (F[0, p[0][i]] - F[0, max_label]).cpu().detach().numpy()/np.linalg.norm((cur_grad - grad_ori).flatten())
    #         # cur_grad = copy.deepcopy(purtabed_img.grad.data)
    #         # cur_grad = torch.autograd.grad(F[0, p[0][i]], purtabed_img,
    #         #                                    grad_outputs=torch.ones_like(F[0, p[0][i]]),
    #         #                                    retain_graph=True)[0]
    #         # norm = torch.div(1,torch.norm(purtabed_img.grad.data - grad_ori))
    #         # w = w + (cur_grad - grad_ori) / norm
    #         # f = f + (F[0, p[0][i]] - F[0, max_label])/ norm
    #         # w = w + (cur_grad - grad_ori)*norm
    #         # f = f + (F[0, p[0][i]] - F[0, max_label])*norm
    #         # del norm
    #         # del cur_grad
    #     # del grad_ori
    #     # del purtabed_img
    #     modifier = modifier + torch.from_numpy(np.float32(abs(f)*w/np.linalg.norm(w))).cuda()
    #     # modifier = modifier + torch.abs(f) * w / torch.norm(w)
    #     # del purtabed_img
    #     purtabed_img = Variable((torch.tanh(modifier + timg) * boxmul + boxplus), requires_grad=True).to(device)
    #     F = model(purtabed_img)
    #     # p = F.cpu().detach().numpy().flatten().argsort()[::-1]
    #     p = torch.argsort(F, dim=1, descending=True)
    #
    #     # complement_set = all_list - set(p[:k_value])
    #     complement_set = all_list - set((p[0][:k_value]).cpu().detach().numpy())
    #     # Flag, predict_label = nontargeted_TP_index(tlab.cpu().detach().numpy(), F.cpu().detach().numpy(), k_value)
    #     if complement_set.issuperset(set(tlab_all)):
    #         Flag, predict_label = nontargeted_TP_index(tlab.cpu().detach().numpy(), F.cpu().detach().numpy(), k_value)
    #         purtabed_img_out = np.arctanh(((purtabed_img - boxplus) / boxmul * 0.999999).cpu().detach().numpy())
    #         modifier_out = purtabed_img_out - timg.cpu().detach().numpy()
    #         attack_success = True
    #         c_list[0], c_list[1], c_list[2], c_list[3], c_list[4], c_list[5] = \
    #             topk_acc_metric_1_to_10(GT.cpu().detach().numpy(), F.cpu().detach().numpy())
    #         print('iter:', loop_i + 1,\
    #               'attacked: ', attack_success, \
    #               'predict_label:', predict_label, \
    #               'GT:', tlab.cpu().detach().numpy(), \
    #               'min:', "{:.5f}".format(modifier.min().cpu()), \
    #               'max:', "{:.5f}".format(modifier.max().cpu()), \
    #               'norm:', "{:.5f}".format(np.linalg.norm(modifier.cpu().detach().numpy())))
    #     loop_i += 1
    # del purtabed_img
    return purtabed_img_out, modifier_out, attack_success, c_list

def kUAP(args, model, device, val_loader, sample_list, train_index_end):
    global_delta = torch.zeros(iter(val_loader).next()[0].shape).to(device)
    fooling_rate = 0
    itr = 0
    boxmul = (args.boxmax - args.boxmin) / 2.
    boxplus = (args.boxmin + args.boxmax) / 2.
    while fooling_rate < args.ufr_lower_bound and itr < args.max_iter_uni:
        print('Starting pass number ', itr)
        index1 =0
        for ith, (data, GT) in tqdm(enumerate(val_loader)):
            if ith in sample_list[:train_index_end]:
                if len(GT.shape) == 3:
                    GT = GT.max(dim=1)[0]
                else:
                    pass
                print('ith:', index1 + 1)
                GT = GT.int()
                print('\n')
                data, GT = data.to(device), GT.to(device)
                pred = model(torch.tanh(data + global_delta) * boxmul + boxplus)
                Flag = bool(topk_acc_metric(GT.cpu().numpy(), pred.cpu().detach().numpy(), args.k_value))
                if Flag == False:
                    purtabed_img_out, modifiered, attack_success, c_list = \
                        kfool(model, data, GT, args.k_value, args.boxmax, args.boxmin, args.maxiter, device=device,
                              lr=args.lr_attack)
                    if attack_success == True:
                        global_delta = global_delta + torch.from_numpy(modifiered).to(device)
                        print('pre_global_delta:', np.linalg.norm(global_delta.cpu().detach().numpy()))
                        print(args.uap_norm, args.uap_eps)
                        global_delta = clip_eta(global_delta, args.uap_norm, args.uap_eps)
                        print('after_global_delta:', np.linalg.norm(global_delta.cpu().detach().numpy()))
                else:
                    print('attack had successed')

                index1 = index1 + 1
            if index1 == train_index_end:
                break
        itr = itr + 1
        count = 0
        data_num = 0
        index2 = 0
        for ith, (data, GT) in tqdm(enumerate(val_loader)):
            if ith in sample_list[:train_index_end]:
                if len(GT.shape) == 3:
                    GT = GT.max(dim=1)[0]
                else:
                    pass
                data, GT = data.to(device), GT.to(device)
                data_num = data_num + 1
                pred = model(torch.tanh(data + global_delta) * boxmul + boxplus)
                count = count + topk_acc_metric(GT.cpu().numpy(), pred.cpu().detach().numpy(), args.k_value)
                index2 = index2 + 1
            if index2 == train_index_end:
                break

        fooling_rate = count / data_num
        print('FOOLING RATE = ', fooling_rate)
    return global_delta


def save_result(success_c_list, index, args, success_img_index, success_modifier_norm_list, success_modifier_norm_index_0, success_modifier_norm_index_1, \
                success_modifier_norm_index_2, success_modifier_norm_index_3,success_modifier_norm_index_4, success_modifier_norm_index_5,\
                success_blurring_index_1, success_blurring_index_2, success_blurring_index_3):
    final_all = []
    c_list_sum = np.sum(np.asarray(success_c_list), 0) / index
    if success_c_list != []:
        print('attack_type= ', "{}".format(args.app), 'label_difficult= ', "{}".format(args.label_difficult))
        print('FR_1= ', "{:.5f}".format(c_list_sum[0]), \
              'FR_2= ', "{:.5f}".format(c_list_sum[1]), \
              'FR_3= ', "{:.5f}".format(c_list_sum[2]), \
              'FR_4= ', "{:.5f}".format(c_list_sum[3]), \
              'FR_5= ', "{:.5f}".format(c_list_sum[4]), \
              'FR_10= ', "{:.5f}".format(c_list_sum[5]))
        print('avg_norm_1= ', "{}".format(
            np.average(np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_0))))), \
              'avg_norm_2= ', "{}".format(np.average(
                np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_1))))), \
              'avg_norm_3= ', "{}".format(np.average(
                np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_2))))), \
              'avg_norm_4= ', "{}".format(np.average(
                np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_3))))), \
              'avg_norm_5= ', "{}".format(np.average(
                np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_4))))), \
              'avg_norm_10= ', "{}".format(np.average(
                np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_5))))))
    else:
        print('All False')

    if success_blurring_index_1!= []:
        success_blurring_ratio_1 = len(success_blurring_index_1)/ index
    else:
        success_blurring_ratio_1 = 0
    if success_blurring_index_2 != []:
        success_blurring_ratio_2 = len(success_blurring_index_2) / index
    else:
        success_blurring_ratio_2 = 0
    if success_blurring_index_3 != []:
        success_blurring_ratio_3 = len(success_blurring_index_3) / index
    else:
        success_blurring_ratio_3 = 0
    print('blurring_1_ratio = {:.5f}'.format(success_blurring_ratio_1),\
          'blurring_2_ratio = {:.5f}'.format(success_blurring_ratio_2),\
          'blurring_3_ratio = {:.5f}'.format(success_blurring_ratio_3))

    # final_all.append(success_perturbated_img_list)
    # final_all.append(success_modifier_list)
    final_all.append(success_modifier_norm_list)
    final_all.append(success_img_index)
    final_all.append(success_c_list)
    final_all.append(success_modifier_norm_index_0)
    final_all.append(success_modifier_norm_index_1)
    final_all.append(success_modifier_norm_index_2)
    final_all.append(success_modifier_norm_index_3)
    final_all.append(success_modifier_norm_index_4)
    final_all.append(success_modifier_norm_index_5)
    final_all.append(success_blurring_index_1)
    final_all.append(success_blurring_index_2)
    final_all.append(success_blurring_index_3)
    os.makedirs('./result/{}/{}/{}/eps_{}'.format(args.dataset, args.label_difficult, args.app, args.eps), exist_ok=True)
    final_all = np.asarray(final_all, dtype=object)
    np.save('./result/{}/{}/{}/eps_{}/k_{}'.format(args.dataset, args.label_difficult, args.app, args.eps, args.k_value), final_all)

def baselineap(args, model, device, val_loader):
    print('kvalue: ',args.k_value, 'label_difficult',args.label_difficult, 'app_type:', args.app,\
          'uap_norm:', args.uap_norm, 'uap_eps:', args.uap_eps)
    model.eval()
    success_count = 0
    index = 0
    index_success = 0
    # success_perturbated_img_list = []
    # success_modifier_list = []
    success_modifier_norm_list = []
    success_img_index = []
    success_c_list = []
    success_modifier_norm_index_0 = []
    success_modifier_norm_index_1 = []
    success_modifier_norm_index_2 = []
    success_modifier_norm_index_3 = []
    success_modifier_norm_index_4 = []
    success_modifier_norm_index_5 = []
    success_blurring_index_1 = []
    success_blurring_index_2 = []
    success_blurring_index_3 = []
    sample_list = np.load('ap_{}_list.npy'.format(args.dataset))

    if args.app == 'baseline_kUAP':
        ###Train
        global_delta = kUAP(args, model, device, val_loader, sample_list, args.uap_train_index_end)
        np.save('./result/{}/{}/{}/eps_{}/perturbation_{}.npy'.format(args.dataset, args.label_difficult, args.app,
                                                                      args.eps, args.k_value),
                global_delta.cpu().detach().numpy())
        ###Test
        boxmul = (args.boxmax - args.boxmin) / 2.
        boxplus = (args.boxmin + args.boxmax) / 2.
        for ith, (data, GT) in tqdm(enumerate(val_loader)):
            if ith in sample_list[args.uap_test_index_start:args.uap_test_index_end]:
                if len(GT.shape) == 3:
                    GT = GT.max(dim=1)[0]
                else:
                    pass
                print('ith2:', index + 1)
                data, GT = data.to(device), GT.to(device)
                pred = model(torch.tanh(data + global_delta) * boxmul + boxplus)
                # Flag, predict_label = nontargeted_TP_index(GT.cpu().numpy(), pred.cpu().detach().numpy(), args.k_value)
                Flag = bool(topk_acc_metric(GT.cpu().numpy(), pred.cpu().detach().numpy(), args.k_value))
                if Flag == True:
                    success_count = success_count + 1
                    c_list = [0] * 6
                    c_list[0], c_list[1], c_list[2], c_list[3], c_list[4], c_list[5] = \
                        topk_acc_metric_1_to_10(GT.cpu().detach().numpy(), pred.cpu().detach().numpy())
                    success_modifier_norm_list.append(np.linalg.norm(global_delta.cpu().detach().numpy())/((args.image_size)*(args.image_size)))
                    success_img_index.append(ith)
                    success_c_list.append(c_list)
                    if c_list[0] == 1:
                        success_modifier_norm_index_0.append(index_success)
                    if c_list[1] == 1:
                        success_modifier_norm_index_1.append(index_success)
                    if c_list[2] == 1:
                        success_modifier_norm_index_2.append(index_success)
                    if c_list[3] == 1:
                        success_modifier_norm_index_3.append(index_success)
                    if c_list[4] == 1:
                        success_modifier_norm_index_4.append(index_success)
                    if c_list[5] == 1:
                        success_modifier_norm_index_5.append(index_success)
                    index_success = index_success + 1
                print('success:{}/{}'.format(success_count, index + 1))
                index = index + 1
            if index == (args.uap_test_index_end - args.uap_test_index_start):
                break

    if args.app == 'baseline_kfool':
        for ith, (data, GT) in tqdm(enumerate(val_loader)):
            if ith in sample_list[:1000]:
                if len(GT.shape) == 3:
                    GT = GT.max(dim=1)[0]
                else:
                    pass
                GT = GT.int()
                if index < 1000:
                    print('\n')
                    data, GT = data.to(device), GT.to(device)
                    purtabed_img_out, modifier_out, attack_success, c_list =\
                        kfool(model, data, GT, args.k_value, args.boxmax, args.boxmin, args.maxiter, device=device, lr=args.lr_attack)



                    if attack_success:
                        ######Gaussian Blurring 3,5,7
                        img_1_ori = cv2.GaussianBlur(purtabed_img_out[0], ksize=(3, 3), sigmaX=0)
                        img_2_ori = cv2.GaussianBlur(purtabed_img_out[0], ksize=(5, 5), sigmaX=0)
                        img_3_ori = cv2.GaussianBlur(purtabed_img_out[0], ksize=(7, 7), sigmaX=0)
                        img_1_ori = np.expand_dims(img_1_ori, axis=0)
                        img_2_ori = np.expand_dims(img_2_ori, axis=0)
                        img_3_ori = np.expand_dims(img_3_ori, axis=0)
                        predict_1 = model(torch.from_numpy(img_1_ori).cuda())
                        Flag_1 = bool(topk_acc_metric(GT.cpu().numpy(), predict_1.cpu().detach().numpy(), args.k_value))
                        predict_2 = model(torch.from_numpy(img_2_ori).cuda())
                        Flag_2 = bool(topk_acc_metric(GT.cpu().numpy(), predict_2.cpu().detach().numpy(), args.k_value))
                        predict_3 = model(torch.from_numpy(img_3_ori).cuda())
                        Flag_3 = bool(topk_acc_metric(GT.cpu().numpy(), predict_3.cpu().detach().numpy(), args.k_value))
                        #############
                        success_count = success_count + 1
                        # success_perturbated_img_list.append(purtabed_img_out)
                        # success_modifier_list.append(modifier_out)
                        success_modifier_norm_list.append(np.linalg.norm(modifier_out)/((args.image_size)*(args.image_size)))
                        success_img_index.append(ith)
                        if Flag_1:
                            success_blurring_index_1.append(ith)
                        if Flag_2:
                            success_blurring_index_2.append(ith)
                        if Flag_3:
                            success_blurring_index_3.append(ith)
                        success_c_list.append(c_list)
                        if c_list[0] == 1:
                            success_modifier_norm_index_0.append(index_success)
                        if c_list[1] == 1:
                            success_modifier_norm_index_1.append(index_success)
                        if c_list[2] == 1:
                            success_modifier_norm_index_2.append(index_success)
                        if c_list[3] == 1:
                            success_modifier_norm_index_3.append(index_success)
                        if c_list[4] == 1:
                            success_modifier_norm_index_4.append(index_success)
                        if c_list[5] == 1:
                            success_modifier_norm_index_5.append(index_success)
                        index_success = index_success + 1

                    print('success:{}/{}'.format(success_count, index + 1))
                    index = index + 1
            if index == 1000:
                break

    if args.app == 'baseline_rank':
        for ith, (data, GT) in tqdm(enumerate(val_loader)):
            if ith in sample_list[:1000]:
                if len(GT.shape) == 3:
                    GT = GT.max(dim=1)[0]
                else:
                    pass
                GT = GT.int()
                if index < 1000:
                    print('\n')
                    data = data.to(device)
                    target, targets_zeros = generate_target_zeros_3_cases(model, data, GT.cpu().detach().numpy(),
                                                                          args.label_difficult, k=args.k_value)
                    targets_zeros = torch.from_numpy(targets_zeros).to(device)

                    ######MLAP rank I baseline from the paper 'Multi-Label Adversarial Perturbations'
                    purtabed_img_out, modifier_out, attack_success, c_list = \
                        rank2(model, data, targets_zeros, args.k_value, args.eps, args.maxiter, args.boxmax, args.boxmin, \
                             GT, device=device, lr=args.lr_attack)
                    if attack_success:
                        success_count = success_count + 1
                        # success_perturbated_img_list.append(purtabed_img_out)
                        # success_modifier_list.append(modifier_out)
                        success_modifier_norm_list.append(np.linalg.norm(modifier_out)/((args.image_size)*(args.image_size)))
                        success_img_index.append(ith)
                        success_c_list.append(c_list)
                        if c_list[0] == 1:
                            success_modifier_norm_index_0.append(index_success)
                        if c_list[1] == 1:
                            success_modifier_norm_index_1.append(index_success)
                        if c_list[2] == 1:
                            success_modifier_norm_index_2.append(index_success)
                        if c_list[3] == 1:
                            success_modifier_norm_index_3.append(index_success)
                        if c_list[4] == 1:
                            success_modifier_norm_index_4.append(index_success)
                        if c_list[5] == 1:
                            success_modifier_norm_index_5.append(index_success)
                        index_success = index_success + 1

                    print('success:{}/{}'.format(success_count, index + 1))

                    index = index + 1
            if index == 1000:
                break

    save_result(success_c_list, index, args, success_img_index, success_modifier_norm_list,
                success_modifier_norm_index_0, success_modifier_norm_index_1, \
                success_modifier_norm_index_2, success_modifier_norm_index_3, \
                success_modifier_norm_index_4, success_modifier_norm_index_5, \
                success_blurring_index_1, success_blurring_index_2, \
                success_blurring_index_3
                )

    torch.cuda.empty_cache()