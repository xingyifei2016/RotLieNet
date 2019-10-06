import torch 
import time, os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

eps = 0.000001

def weightNormalize1(weights):
    return ((weights**2)/torch.sum(weights**2))

def weightNormalize2(weights):
    return weights/torch.sum(weights**2)

def weightNormalize(weights, drop_prob=0.0):
    out = []
    for row in weights:
        if drop_prob==0.0:
            out.append(row**2/torch.sum(row**2))
        else:
            p = torch.randint(0, 2, (row.size())).float().cuda() 
            out.append((row**2/torch.sum(row**2))*p)
    return torch.stack(out)


class ComplexConv2Deffangle(nn.Module):
    def __init__(self, in_channels, out_channels, kern_size, stride, drop_prob=0.0):
        super(ComplexConv2Deffangle, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kern_size = kern_size
        self.stride = stride
        self.drop_prob = drop_prob
        self.weight_matrix_rot1 = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        self.weight_matrix_rot2 = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)
        
    def forward(self, x):
        x_shape = x.shape
        out_spatial_x = int(math.floor((x_shape[3]-(self.kern_size[0]-1)-1)/self.stride[0] + 1))
        out_spatial_y = int(math.floor((x_shape[4]-(self.kern_size[1]-1)-1)/self.stride[1] + 1))
        
        #Shape: [batches, features, in_channels, spatial_x, spatial_y] -> [batches*features, in_channels, spatial_x, spatial_y]
        x = x.view(-1,self.in_channels,x_shape[3],x_shape[4])
        
        #Shape: [batches, features, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets = nn.Unfold(kernel_size=self.kern_size, stride=self.stride)(x).view(x_shape[0], x_shape[1],  self.in_channels, self.kern_size[0]*self.kern_size[1], -1)
        
        #Shape: [batches, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_rot = temporal_buckets[:,0,...]
        
        #Shape: [batches, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_abs = temporal_buckets[:,1,...]
       
        tbr_shape = temporal_buckets_rot.shape 
        
        out_rot = ((torch.sum(temporal_buckets_rot.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*weightNormalize1(self.weight_matrix_rot1),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        out_rot_shape = out_rot.shape
        out_rot = out_rot.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_rot = (torch.sum(out_rot*weightNormalize1(self.weight_matrix_rot2),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        tba_shape = temporal_buckets_abs.shape
        
        #Shape: [batches,  in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_abs = torch.log((temporal_buckets_abs+eps))
        tba_shape = temporal_buckets_abs.shape   
        
        #Shape: [batches, in_channels, L]
        out_abs = ((torch.sum(temporal_buckets_abs.permute(0,3,1,2).contiguous().view(-1, tba_shape[1], tba_shape[2])*weightNormalize(self.weight_matrix_rot1,self.drop_prob),2))).view(tba_shape[0],tba_shape[3],tba_shape[1]).permute(0,2,1).contiguous()
        out_abs_shape = out_abs.shape
        out_abs = out_abs.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        
        #Shape: [batches, 1, out_channels, out_spatial_x, out_spatial_y]
        out_abs = torch.exp(torch.sum(out_abs*weightNormalize(self.weight_matrix_rot2,self.drop_prob),2)).view(out_abs_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        
        return torch.cat((out_rot,out_abs),1)
    
    


class ComplexLinearangle2Dmw_outfield(nn.Module):
    #input_dim should equal channels*frames of previous layer.
    def __init__(self, input_dim):
        super(ComplexLinearangle2Dmw_outfield, self).__init__()
        self.input_dim = input_dim
        self.weight = torch.nn.Parameter(torch.rand([2]), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.rand([2]), requires_grad=True)
        self.weights = torch.nn.Parameter(torch.rand(input_dim), requires_grad=True)

    def ComplexunweightedMeanLinear(self, x_rot, x_abs):
        #x_rot.shape: [batches, out_channels* out_spatial_x* out_spatial_y]
        #x_abs.shape: [batches, out_channels* out_spatial_x* out_spatial_y]
        out_rot = torch.sum(x_rot*weightNormalize1(self.weights),1) * torch.tanh(-self.bias[0])
        x_abs_log = torch.log(x_abs+eps)
        out_abs = torch.exp(torch.sum(x_abs_log*weightNormalize1(self.weights),1))+torch.exp(-self.bias[1]**2)    
        return (out_rot,out_abs)

    def unweightedFMComplex(self, point_list_rot, point_list_abs):
        return self.ComplexunweightedMeanLinear(point_list_rot, point_list_abs)
    
    def Xmetric(self, X, Y):
        return torch.abs(X-Y)
    
    def SOmetric(self, X, Y):
        return torch.abs(X-Y)
       
    def P1metric(self, X, Y):
        return torch.abs(torch.log(X/(Y+eps)))

    def forward(self, x):
        #shape: [batches, out_channels* out_spatial_x* out_spatial_y, 2]
        all_data = x.permute(0,2,3,4,1).contiguous()
        all_data_shape = all_data.shape
        all_data = all_data.view(all_data_shape[0], all_data_shape[1]*all_data_shape[2]*all_data_shape[3], all_data_shape[4])
        all_data_rot = all_data[:,:,0]
        all_data_abs = all_data[:,:,1]
        all_shape = all_data_rot.shape
           
        M_rot, M_abs = self.unweightedFMComplex(all_data_rot, all_data_abs)
              
        dist_rot = self.SOmetric(all_data_rot.view(-1), M_rot.unsqueeze(1).repeat(1,all_shape[1]).view(-1)).view(all_shape[0],all_shape[1])       
        dist_abs = self.P1metric(all_data_abs.view(-1), M_abs.unsqueeze(1).repeat(1,all_shape[1]).view(-1)).view(all_shape[0],all_shape[1])
        dist_l1 = (self.weight[0]**2)*dist_rot + (self.weight[1]**2)*dist_abs
        dist_l1 = dist_l1.view(all_data_shape[0], all_data_shape[1], all_data_shape[2], all_data_shape[3]) 
        return dist_l1




class manifoldReLUv2angle(nn.Module):
    def __init__(self,channels):
        super(manifoldReLUv2angle, self).__init__()
        self.weight_rot = torch.nn.Parameter(torch.rand(1,channels), requires_grad=True)
        self.weight_abs = torch.nn.Parameter(torch.rand(1,channels), requires_grad=True)
        self.channels = channels 

    def forward(self, x):
        #Shape: [batches, features, in_channels, spatial_x, spatial_y]
        x_shape = x.shape  
        temp_rot = x[:,0,...]
        temp_abs = x[:,1,...]  
        temp_rot_prod = (temp_rot.unsqueeze(1)*(weightNormalize2(self.weight_rot+eps)).unsqueeze(0).unsqueeze(3).unsqueeze(4).repeat(x_shape[0],1,1,x_shape[3],x_shape[4]))
        temp_abs = (temp_abs.unsqueeze(1)+(weightNormalize1(self.weight_abs+eps)).unsqueeze(0).unsqueeze(3).unsqueeze(4).repeat(x_shape[0],1,1,x_shape[3],x_shape[4]))
        return torch.cat((temp_rot_prod, temp_abs),1)
    

class ComplexConv2Deffgroup(nn.Module):
      def __init__(self, in_channels, out_channels, kern_size, stride, do_conv=True):
        super(ComplexConv2Deffgroup, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kern_size = kern_size
        self.stride = stride
        self.do_conv = do_conv
        self.wmr = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        self.wma = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True) 
        if do_conv: 
            self.complex_conv = ComplexConv2Deffangle(in_channels, out_channels, kern_size, stride)
        else:
            self.new_wr = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)
            self.new_wa = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)
      
      def ComplexweightedMean(self, x_rot, x_abs):
        x_shape = x_rot.shape
        out_rot = torch.sum(x_rot*weightNormalize1(self.w1), 2).unsqueeze(1).repeat(1, self.out_channels, 1)
        out_rot = torch.sum(out_rot*weightNormalize1(self.w2),2)
        x_abs_log = torch.log(x_abs+eps)
        out_abs = torch.sum(x_abs_log*weightNormalize1(self.w1), 2).unsqueeze(1).repeat(1, self.out_channels, 1)
        out_abs = torch.exp(torch.sum(out_abs*weightNormalize1(self.w2), 2))
        return (out_rot,out_abs)

      def forward(self, x):
        x_shape = x.shape
        out_spatial_x = int(math.floor((x_shape[3]-(self.kern_size[0]-1)-1)/self.stride[0] + 1))
        out_spatial_y = int(math.floor((x_shape[4]-(self.kern_size[1]-1)-1)/self.stride[1] + 1))
        
        #Shape: [batches, features, in_channels, spatial_x, spatial_y] -> [batches*features, in_channels, spatial_x, spatial_y]
        x = x.view(-1,self.in_channels,x_shape[3],x_shape[4])
        
        #Shape: [batches, features, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets = nn.Unfold(kernel_size=self.kern_size, stride=self.stride)(x).view(x_shape[0], x_shape[1],  self.in_channels, self.kern_size[0]*self.kern_size[1], -1)
        
        #Shape: [batches, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_rot = temporal_buckets[:,0,...]
        
        #Shape: [batches, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_abs = temporal_buckets[:,1,...]
        
        tbr_shape0 = temporal_buckets_rot.shape
        temporal_buckets_rot = temporal_buckets_rot.permute(0,3,1,2).contiguous().view(-1, tbr_shape0[1], tbr_shape0[2])
        temporal_buckets_abs = temporal_buckets_abs.permute(0,3,1,2).contiguous().view(-1, tbr_shape0[1],tbr_shape0[2])
       
        #Shape: [batches*L, in_channels, kern_size[0]*kern_size[1]]
        tbr_shape = temporal_buckets_rot.shape 
        in_rot = temporal_buckets_rot * weightNormalize2(self.wmr)
        in_abs = temporal_buckets_abs + weightNormalize1(self.wma)
        if self.do_conv:
            in_rot = in_rot.view(tbr_shape0[0], out_spatial_x, out_spatial_y, -1).permute(0,3,1,2).contiguous().unsqueeze(1)
            in_abs = in_abs.view(tbr_shape0[0], out_spatial_x, out_spatial_y, -1).permute(0,3,1,2).contiguous().unsqueeze(1)
            in_ = torch.cat((in_rot, in_abs), 1).view(tbr_shape0[0], -1, out_spatial_x*out_spatial_y)
            in_fold = nn.Fold(output_size=(x_shape[3],x_shape[4]), kernel_size=self.kern_size, stride=self.stride)(in_)
            in_fold = in_fold.view(x_shape[0],x_shape[1],x_shape[2],x_shape[3],x_shape[4])
            out = self.complex_conv(in_fold)
        else:
            in_rot = torch.mean(in_rot, 2).view(tbr_shape0[0], out_spatial_x, out_spatial_y, -1).permute(0,3,1,2).contiguous().unsqueeze(1)
            in_abs = torch.mean(in_abs, 2).view(tbr_shape0[0], out_spatial_x, out_spatial_y, -1).permute(0,3,1,2).contiguous().unsqueeze(1)
            out = torch.cat((in_rot,in_abs),1)
        return out 
    
    
class ResidualLayer(nn.Module):
    
    def __init__(self, in_channel1, in_channel2, out_channel, kern_size, stride):
        super(ResidualLayer, self).__init__()
        
        #We assume in_channel1 has larger spatial resolution
        self.in_channel1 = in_channel1
        self.in_channel2 = in_channel2
        self.out_channel = out_channel
        self.transform = ComplexConv2Deffgroup(in_channel1, in_channel1, kern_size, stride)
        self.w = torch.nn.Parameter(torch.rand(out_channel, in_channel1+in_channel2), requires_grad=True) 
        
    def forward(self, x1, x2):
        #Assume all square spatial resolutions, x1 has higher resolution
        x1 = self.transform(x1)
        x = torch.cat((x1, x2), 2)
        
        #B, in, H, W
        x_rot = x[:, 0, :, :, :]
        out_rot = x_rot.permute(0, 2, 3, 1).unsqueeze(-2).repeat(1, 1, 1, self.out_channel, 1)
        out_rot = torch.sum(out_rot*weightNormalize1(self.w),-1).unsqueeze(1)
        x_abs = x[:, 1, :, :, :]
        
        x_abs = x_abs.permute(0, 2, 3, 1).unsqueeze(-2).repeat(1, 1, 1, self.out_channel, 1)
        out_abs = torch.log(x_abs+eps)
        out_abs = torch.exp(torch.sum(out_abs*weightNormalize1(self.w), -1)).unsqueeze(1)
       
        return torch.cat((out_rot, out_abs), 1).permute(0, 1, 4, 2, 3)