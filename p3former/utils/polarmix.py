# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Written by Aoran Xiao, 09:43 2022/03/05
# Wish for world peace!

import numpy as np
import torch


def swap(pt1, pt2, start_angle, end_angle, label1, label2, ins_label1, ins_label2):
    # calculate horizontal angle for each point
    yaw1 = -torch.atan2(pt1[:, 1], pt1[:, 0])
    yaw2 = -torch.atan2(pt2[:, 1], pt2[:, 0])

    # select points in sector
    idx1 = (yaw1>start_angle) & (yaw1<end_angle)
    idx2 = (yaw2>start_angle) & (yaw2<end_angle)

    # swap
    pt1_out = pt1[~idx1]
    pt1_out = torch.cat((pt1_out, pt2[idx2]))
    pt2_out = pt2[~idx2]
    pt2_out = torch.cat((pt2_out, pt1[idx1]))

    label1_out = label1[~idx1]
    label1_out = torch.cat((label1_out, label2[idx2]))
    label2_out = label2[~idx2]
    label2_out = torch.cat((label2_out, label1[idx1]))

    ins_label1_out = ins_label1[~idx1]
    ins_label1_out = torch.cat((ins_label1_out, ins_label2[idx2]))
    ins_label2_out = ins_label2[~idx2]
    ins_label2_out = torch.cat((ins_label2_out, ins_label1[idx1]))

    assert pt1_out.shape[0] == label1_out.shape[0]
    assert pt2_out.shape[0] == label2_out.shape[0]

    return pt1_out, pt2_out, label1_out, label2_out, ins_label1_out, ins_label2_out

def rotate_copy(pts, labels, instance_classes, Omega, ins_labels):
    # extract instance points
    pts_inst, labels_inst, ins_labels_inst = [], [], []
    for s_class in instance_classes:
        pt_idx = labels == s_class
        pts_inst.append(pts[pt_idx])
        labels_inst.append(labels[pt_idx])
        ins_labels_inst.append(ins_labels[pt_idx])
    pts_inst = torch.cat(pts_inst, axis=0)
    labels_inst = torch.cat(labels_inst, axis=0)
    ins_labels_inst = torch.cat(ins_labels_inst, axis=0)

    # rotate-copy
    pts_copy = [pts_inst]
    labels_copy = [labels_inst]
    ins_labels_copy = [ins_labels_inst]

    for omega_j in Omega:
        rot_mat = pts_inst.new_tensor([[np.cos(omega_j),
                             np.sin(omega_j), 0],
                            [-np.sin(omega_j),
                             np.cos(omega_j), 0], [0, 0, 1]])
        new_pt = pts_inst.new_zeros(pts_inst.shape)
        new_pt[:, :3] = torch.matmul(pts_inst[:, :3], rot_mat)
        new_pt[:, 3] = pts_inst[:, 3]
        pts_copy.append(new_pt)
        labels_copy.append(labels_inst)
        ins_labels_copy.append(ins_labels_inst)

    pts_copy = torch.cat(pts_copy, axis=0)
    labels_copy = torch.cat(labels_copy, axis=0)
    ins_labels_copy = torch.cat(ins_labels_copy, axis=0)

    return pts_copy, labels_copy, ins_labels_copy

def polarmix(pts1, labels1, ins_labels1, pts2, labels2, ins_labels2, alpha, beta, instance_classes, Omega, swapping_prob, pasting_prob):
    ins_labels2 += 10000<<16
    pt1_out, label1_out, ins_label1_out, pt2_out, label2_out, ins_label2_out = pts1, labels1, ins_labels1, pts2, labels2, ins_labels2
    # swapping
    if np.random.random() < swapping_prob:
        pt1_out, pt2_out, label1_out, label2_out, ins_label1_out, ins_label2_out  = swap(pts1, pts2, start_angle=alpha, end_angle=beta,
                label1=labels1, label2=labels2, ins_label1=ins_labels1, ins_label2=ins_labels2)

    # rotate-pasting
    if np.random.random() < pasting_prob:
        # rotate-copy
        pts_copy1, labels_copy1, ins_labels_copy1 = rotate_copy(pts2, labels2, instance_classes, Omega, ins_labels2)
        pts_copy2, labels_copy2, ins_labels_copy2 = rotate_copy(pts1, labels1, instance_classes, Omega, ins_labels1)
        # paste
        pt1_out = torch.cat((pt1_out, pts_copy1), axis=0)
        label1_out = torch.cat((label1_out, labels_copy1), axis=0)
        ins_label1_out = torch.cat((ins_label1_out, ins_labels_copy1), axis=0)
        pt2_out = torch.cat((pt2_out, pts_copy2), axis=0)
        label2_out = torch.cat((label2_out, labels_copy2), axis=0)
        ins_label2_out = torch.cat((ins_label2_out, ins_labels_copy2), axis=0)

    return pt1_out, pt2_out, label1_out, label2_out, ins_label1_out, ins_label2_out