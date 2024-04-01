# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from numpy.core.numeric import False_
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
import numpy as np
import numba as nb
import torch_scatter
import time
from p3former.utils.polarmix import polarmix

class SphericalVoxelization(nn.Module):
    def __init__(self,
                 point_cloud_range,
                 grid_size,
                 thing_list,
                 trans_std=[0.1,0.1,0.1],
                 center_type='Axis_center',
                 instance_aug=False,
                 instance_aug_cfg=dict(
                     path='data/instance_path.pkl',
                     inst_global_aug=True,
                     inst_loc_aug=True,
                     inst_os=True,
                    ),
                rotate=True,
                flip=True,
                scale=True,
                noise=True,
                add_num=5,
                use_polarmix=False,
                use_copymix=False,
                mix_aug=False,
                polar_swap_prob=0.5,
                polar_past_prob=0.5,
                laser_mix_prob=0.5,
                copy_mix_prob=0.5,
                 ):
        super(SphericalVoxelization, self).__init__()
        """
        Args:
            voxel_size (list): list [x, y, z] size of three dimension
            point_cloud_range (list):
                [x_min, y_min, z_min, x_max, y_max, z_max]
            max_num_points (int): max number of points per voxel
            max_voxels (tuple or int): max number of voxels in
                (training, testing) time
            deterministic: bool. whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.
        """
        self.point_cloud_range = point_cloud_range
        self.class_num = 20
        self.list_point_cloud_range = point_cloud_range
        if point_cloud_range[1]=='-np.pi':
            point_cloud_range[1]=-np.pi
            point_cloud_range[4]=np.pi

        self.point_cloud_range = torch.tensor(
            self.point_cloud_range).double().cuda()
        # [0, -40, -3, 70.4, 40, 1]
        self.list_grid_size = grid_size
        self.grid_size = torch.tensor(grid_size).double().cuda()
        input_feat_shape = grid_size[:2]
        # the origin shape is as [x-len, y-len, z-len]
        # [w, h, d] -> [d, h, w]
        self.pcd_shape = [*input_feat_shape, 1][::-1]
        self.ignore_label = 0
        self.return_test = False
        self.trans_std = trans_std
        assert center_type in ['Axis_center', 'Mass_center']
        self.center_type = center_type
        self.instance_aug = instance_aug
        self.rotate = rotate
        self.flip = flip
        self.scale = scale
        self.noise = noise

        self.use_polarmix = use_polarmix
        self.use_copymix = use_copymix

        self.polar_swap_prob = polar_swap_prob
        self.polar_past_prob = polar_past_prob
        self.laser_mix_prob = laser_mix_prob
        self.copy_mix_prob = copy_mix_prob
        self.mix_aug = mix_aug
        self.thing_list = thing_list

    def forward(self,data,aug=True):
        data_tuple=[]
        if len(data) == 1:
            for xyz in data[0]:
                data_tuple.append(self.forward_test([xyz[:,:3],xyz[:,3]]))
            data_tuple = self.collate_test(data_tuple)
        elif len(data) == 4:
            if aug:
                if self.use_polarmix:
                    instance_classes = self.thing_list
                    Omega = [np.random.random() * np.pi * 2 / 3, (np.random.random() + 1) * np.pi * 2 / 3]  # x3
                    for idx in range(int(len(data[0])/2)):
                        alpha = (np.random.random() - 1) * np.pi
                        beta = alpha + np.pi
                        pt1_out, pt2_out, label1_out, label2_out, ins_label1_out, ins_label2_out = polarmix(data[0][2*idx],data[1][2*idx],data[2][2*idx],data[0][2*idx+1],data[1][2*idx+1],data[2][2*idx+1],alpha=alpha, beta=beta,
                                                                                        instance_classes=instance_classes,
                                                                                        Omega=Omega, swapping_prob=self.polar_swap_prob, pasting_prob=self.laser_mix_prob)
                        data[0][2*idx]=pt1_out
                        data[1][2*idx]=label1_out
                        data[2][2*idx]=ins_label1_out
                        data[0][2*idx+1]=pt2_out
                        data[1][2*idx+1]=label2_out
                        data[2][2*idx+1]=ins_label2_out

                if self.use_copymix:
                    for idx in range(int(len(data[0]))):
                        pt_out, label_out, ins_label_out = copymix(data[0][idx],data[1][idx],data[2][idx], mix_prob=self.copy_mix_prob, thing_list=self.thing_list)
                        data[0][idx]=pt_out
                        data[1][idx]=label_out
                        data[2][idx]=ins_label_out

                if self.mix_aug:
                    prob = np.random.choice(2, 1)
                    if prob == 1:
                        for idx in range(int(len(data[0])/2)):
                            pt1_out, pt2_out, label1_out, label2_out, ins_label1_out, ins_label2_out = lasermix_aug(data[0][2*idx],data[1][2*idx],data[2][2*idx],data[0][2*idx+1],data[1][2*idx+1],data[2][2*idx+1])

                            data[0][2*idx]=pt1_out
                            data[1][2*idx]=label1_out
                            data[2][2*idx]=ins_label1_out
                            data[0][2*idx+1]=pt2_out
                            data[1][2*idx+1]=label2_out
                            data[2][2*idx+1]=ins_label2_out
                    
                    elif prob == 0:
                        instance_classes = self.thing_list
                        Omega = [np.random.random() * np.pi * 2 / 3, (np.random.random() + 1) * np.pi * 2 / 3]  # x3
                        for idx in range(int(len(data[0])/2)):
                            alpha = (np.random.random() - 1) * np.pi
                            beta = alpha + np.pi
                            pt1_out, pt2_out, label1_out, label2_out, ins_label1_out, ins_label2_out = polarmix(data[0][2*idx],data[1][2*idx],data[2][2*idx],data[0][2*idx+1],data[1][2*idx+1],data[2][2*idx+1],alpha=alpha, beta=beta,
                                                                                            instance_classes=instance_classes,
                                                                                            Omega=Omega, swapping_prob=self.polar_swap_prob, pasting_prob=self.laser_mix_prob)
                            data[0][2*idx]=pt1_out
                            data[1][2*idx]=label1_out
                            data[2][2*idx]=ins_label1_out
                            data[0][2*idx+1]=pt2_out
                            data[1][2*idx+1]=label2_out
                            data[2][2*idx+1]=ins_label2_out

                for xyz, labels, inst_labels, valid in zip(data[0],data[1],data[2],data[3]):
                    data_tuple.append(self.forward_train([xyz[:,:3],labels,xyz[:,3], inst_labels, valid],aug=False)) #TODO better way for insert sig
                data_tuple = self.collate_train(data_tuple)

        return data_tuple

    def collate_train(self,data):
        #data2stack = torch.stack([d[0] for d in data]).astype(np.float32)
        vox_label = torch.stack([d['processed_label'] for d in data])                # grid-wise  semantic label
        grid_ind_stack = [d['grid_ind'] for d in data]                        # grid-wise  grid coor
        point_label = [d['labels'] for d in data]                           # point-wise semantic label
        pt_fea = [d['return_fea'].float().cuda() for d in data]                 # point-wise feature
        pt_ins_labels = [d['inst_labels'] for d in data]                         # point-wise instance label
        #pt_offsets = [d['offsets'] for d in data]                            # point-wise center offset
        pt_valid = [d['valid'] for d in data]                              # point-wise indicator for foreground points
        pt_cart_xyz = [d['xyz'] for d in data]                           # point-wise cart coor
        sparse_sem_labels = [d['max_labels'] for d in data]                     # sparse grid-wise semantic label
        sparse_inst_labels = [d['max_inst_labels'] for d in data]                   # sparse grid-wise instance label
        #sparse_inst_offsets = [d['grid_offset'] for d in data]   
        labels_coor = [d['labels_coor'] for d in data]
        grid_size = [d['grid_size'] for d in data]
        sort_inst_labels = [d['sort_inst_labels'] for d in data]
        # center_cls = [d[13] for d in data]
        return {
            'vox_coor':None,
            'vox_label': vox_label.long().cuda(),#loss need long
            'grid': grid_ind_stack,
            'pt_labs': point_label,
            'pt_fea': pt_fea,
            'pt_ins_labels': pt_ins_labels,
            #'pt_offsets': pt_offsets,
            'pt_valid': pt_valid,
            'pt_cart_xyz': pt_cart_xyz,
            'sparse_sem_labels': sparse_sem_labels,
            'sparse_inst_labels': sparse_inst_labels,
            #'sparse_inst_offsets': sparse_inst_offsets,
            'labels_coor': labels_coor,
            'grid_size': grid_size,
            'sort_inst_labels': sort_inst_labels,
            # 'center_cls': center_cls,
        }
    
    def collate_test(self,data):
        #data2stack = torch.stack([d[0] for d in data]).astype(np.float32)
        grid_ind_stack = [d[1].type(torch.LongTensor).cuda() for d in data]
        xyz = [d[2].float().cuda() for d in data]
        pt_cart_xyz = [d[3] for d in data]
        return {
            'grid': grid_ind_stack,
            'pt_fea': xyz,
            'pt_cart_xyz': pt_cart_xyz,
        }

    def forward_train(self, data, aug=True): 

        if len(data) == 5:
            xyz, labels, sig, inst_labels, valid, = data
        else:
            raise Exception('Return invalid data tuple')

        #aug = False # debug
        # print(xyz[labels>0].max())
        if aug:
            #aug
            # random data augmentation by rotation
            if self.rotate:
                rotate_rad = np.deg2rad(np.random.random() * 90) - np.pi / 4
                #rotate_rad = np.deg2rad(np.random.random()*360)# differen from cylinder3d
                c, s = np.cos(rotate_rad), np.sin(rotate_rad)
                j = np.matrix([[c, s], [-s, c]])
                j = torch.from_numpy(j).float().cuda()
                xyz[:, :2] = torch.matmul(xyz[:, :2], j)

            # random data augmentation by flip x , y or x+y
            if self.flip:
                flip_type = np.random.choice(4, 1)
                if flip_type == 1:
                    xyz[:, 0] = -xyz[:, 0]
                elif flip_type == 2:
                    xyz[:, 1] = -xyz[:, 1]
                elif flip_type == 3:
                    xyz[:, :2] = -xyz[:, :2]

            # random data augmentation by scale
            if self.scale:
                noise_scale = torch.Tensor(1).uniform_(0.95, 1.05).cuda()
                xyz[:, 0] = noise_scale * xyz[:, 0]
                xyz[:, 1] = noise_scale * xyz[:, 1]

            # random data augmentation by noise
            if self.noise: 
                noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                                np.random.normal(0, self.trans_std[1], 1),
                                                np.random.normal(0, self.trans_std[2], 1)]).T
                noise_translate = torch.from_numpy(noise_translate).cuda()
                xyz[:, :3] += noise_translate

        # import time
        # t1 = time.time()

            if self.instance_aug:
                xyz = xyz.cpu().numpy()
                labels = labels.cpu().numpy()
                inst_labels = inst_labels.cpu().numpy()
                sig = sig.cpu().numpy()

                #t2 = time.time()
                xyz,labels,inst_labels,sig = self.inst_aug.instance_aug(xyz,labels.squeeze(),inst_labels.squeeze(),sig[:,None])
                #t3 = time.time()
                xyz = torch.from_numpy(xyz).cuda()
                labels = torch.from_numpy(labels).cuda().squeeze()
                inst_labels = torch.from_numpy(inst_labels).cuda().squeeze()
                sig = torch.from_numpy(sig).cuda().squeeze()      
        
        # t4 = time.time()
        # print('t1:',t2-t1)
        # print('t2:',t3-t2)
        # print('t3:',t4-t3)

        #transform to polar
        xyz_pol = self.cart2polar(xyz)

        max_bound = self.point_cloud_range[3:]
        min_bound = self.point_cloud_range[:3]
        crop_range = max_bound - min_bound
        voxel_size = crop_range/(self.grid_size-1)# (size-1) could directly get index starting from 0, very convenient

        grid_ind = xyz_pol.new_zeros(size=(xyz_pol.size(0), 3), dtype=torch.long)        
        grid_ind = torch.floor((self.torch_clip(xyz_pol.clone(),min_bound,max_bound)- min_bound) / voxel_size).long().cuda()# Cylinder3DInstanceHead need long


        #get return_fea
        voxel_centers = (grid_ind.float() + 0.5) * voxel_size + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = torch.cat((return_xyz, xyz_pol, xyz[:, :2]), axis=1)#N×3,N×3,N×2
        return_fea = torch.cat((return_xyz, torch.unsqueeze(sig,1)), axis=1)

        #get grid-wise label
        processed_label, max_labels, max_inst_labels, labels_coor, sort_inst_labels= self.precise_aggregate_label(grid_ind,labels,inst_labels)

        data_tuple = (None,processed_label)
        data_tuple = dict()
        data_tuple['processed_label'] = processed_label
        data_tuple['grid_ind'] = grid_ind
        data_tuple['labels'] = labels
        data_tuple['return_fea'] = return_fea
        data_tuple['inst_labels'] = inst_labels
        data_tuple['valid'] = valid
        data_tuple['xyz'] = xyz
        data_tuple['max_labels'] = max_labels
        data_tuple['max_inst_labels'] = max_inst_labels
        data_tuple['labels_coor'] = labels_coor
        data_tuple['grid_size'] = self.grid_size
        data_tuple['sort_inst_labels'] = sort_inst_labels

        return data_tuple

    def forward_test(self, data):

        if len(data) ==2:
            xyz, sig = data
            xyz_pol = self.cart2polar(xyz)
        else:
            raise Exception('Return invalid data tuple')

        max_bound = self.point_cloud_range[3:]
        min_bound = self.point_cloud_range[:3]
        crop_range = max_bound - min_bound
        voxel_size = crop_range/(self.grid_size-1)

        grid_ind = xyz_pol.new_zeros(size=(xyz_pol.size(0), 3), dtype=torch.long)
        
        grid_ind = torch.floor((self.torch_clip(xyz_pol.clone(),min_bound,max_bound)-min_bound) / voxel_size).int().cuda()#50-3=47除以interval恰好在一个边缘

        cur_grid_size = self.grid_size

        intervals = crop_range / (cur_grid_size - 1) 
        voxel_centers = (grid_ind.float() + 0.5) * intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = torch.cat((return_xyz, xyz_pol, xyz[:, :2]), axis=1)#N×3,N×3,N×2

        return_fea = torch.cat((return_xyz, torch.unsqueeze(sig,1)), axis=1)

        if self.return_test:
            data_tuple = (None, grid_ind, return_fea, 0, xyz)
        else:
            data_tuple = (None, grid_ind, return_fea, xyz)
        return data_tuple

    def cart2polar(self, input_xyz):
        rho = torch.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
        phi = torch.atan2(input_xyz[:, 1], input_xyz[:, 0])
        return torch.stack((rho, phi, input_xyz[:, 2]), axis=1)
    
    def torch_clip(self,src,rmin,rmax):
        for i in range(3):
            assert src.shape[1]==3
            assert rmin.shape[0]==3
            assert rmax.shape[0]==3

            src[:,i] = torch.clamp(src[:,i],rmin[i],rmax[i])
        return src

    def aggregate_label(self, grid_ind, labels, inst_labels):
        pt_num = grid_ind.shape[0]
        
        unq, unq_inv, unq_cnt = torch.unique(grid_ind, return_inverse=True, return_counts=True, dim=0)
        unq2tuple = (unq[:,0].long(),unq[:,1].long(),unq[:,2].long())
        
        # process label
        flatten_label = grid_ind.new_zeros((pt_num,self.class_num))
        flatten_label[list(range(0, pt_num)),labels[list(range(0, pt_num))].cpu().numpy().tolist()] +=1
        flatten_label_sum = torch_scatter.scatter_sum(flatten_label, unq_inv, dim=0)
        max_labels = torch.argmax(flatten_label_sum,dim=1)
        processed_label = grid_ind.new_ones(self.list_grid_size) * self.ignore_label
        processed_label = processed_label.long()
        processed_label.index_put_(unq2tuple,max_labels)

        unq_inst_labels, unq_inv_inst, unq_cnt_inst = torch.unique(inst_labels, return_inverse=True, return_counts=True, dim=0)
        sort_inst_labels = inst_labels.new_zeros(inst_labels.shape)
        i=1

        for u in unq_inst_labels:
            valid = inst_labels == u
            a = labels[valid]
            assert (a-a[0]).sum()==0
        
        for u in unq_inst_labels:
            #if u & 0xFFFF in valid_xentropy_ids:
            valid = inst_labels == u
            sort_inst_labels[valid] = i
            i = i + 1
        
        flatten_inst_label = grid_ind.new_zeros((pt_num,i)) 
        flatten_inst_label[list(range(0, pt_num)),sort_inst_labels[list(range(0, pt_num))].cpu().numpy().tolist()] +=1
        flatten_inst_label_sum = torch_scatter.scatter_sum(flatten_inst_label, unq_inv, dim=0)
        max_inst_labels = torch.argmax(flatten_inst_label_sum,dim=1)

        return processed_label, max_labels, max_inst_labels, unq, sort_inst_labels 

    def precise_aggregate_label(self, grid_ind, labels, inst_labels):
        pt_num = grid_ind.shape[0]
        
        unq, unq_inv, unq_cnt = torch.unique(grid_ind, return_inverse=True, return_counts=True, dim=0)
        unq2tuple = (unq[:,0].long(),unq[:,1].long(),unq[:,2].long())
        
        unq_inst_labels, unq_inv_inst, unq_cnt_inst = torch.unique(inst_labels, return_inverse=True, return_counts=True, dim=0)
        sort_inst_labels = inst_labels.new_zeros(inst_labels.shape)
        i=1

        
        ins2sem = labels.new_zeros(size=[len(unq_inst_labels)])
        for u in unq_inst_labels:
            valid = inst_labels == u
            sort_inst_labels[valid] = i
            ins2sem[i-1] = labels[valid][0]
            i = i + 1
        
        flatten_inst_label = grid_ind.new_zeros((pt_num,i)) 
        flatten_inst_label[list(range(0, pt_num)),sort_inst_labels[list(range(0, pt_num))].cpu().numpy().tolist()] +=1
        flatten_inst_label_sum = torch_scatter.scatter_sum(flatten_inst_label, unq_inv, dim=0)
        max_inst_labels = torch.argmax(flatten_inst_label_sum,dim=1)

        max_labels = ins2sem[max_inst_labels-1]

        processed_label = grid_ind.new_ones(self.list_grid_size) * self.ignore_label
        processed_label = processed_label.long()
        processed_label.index_put_(unq2tuple,max_labels)

        return processed_label, max_labels, max_inst_labels, unq, sort_inst_labels 


    def get_center(self, inst_labels, labels_coor):
        label_unique = inst_labels.unique()
        grid_offset = inst_labels.new_zeros(size=[inst_labels.shape[0],3]).float()
        for i in label_unique:
            mask = inst_labels == i
            inst_center = self.calc_xyz_middle(labels_coor[mask]) 
            grid_offset[mask] = labels_coor[mask] - inst_center
        return grid_offset 
    
    def calc_xyz_middle(self,xyz):
        return torch.stack([
            (torch.max(xyz[:, 0]) + torch.min(xyz[:, 0])) / 2.0,
            (torch.max(xyz[:, 1]) + torch.min(xyz[:, 1])) / 2.0,
            (torch.max(xyz[:, 2]) + torch.min(xyz[:, 2])) / 2.0
            ])

def calc_xyz_middle(xyz):
    return np.array([
        (np.max(xyz[:, 0]) + np.min(xyz[:, 0])) / 2.0,
        (np.max(xyz[:, 1]) + np.min(xyz[:, 1])) / 2.0,
        (np.max(xyz[:, 2]) + np.min(xyz[:, 2])) / 2.0
    ], dtype=np.float32)
