import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

### torch version too old for timm
### https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers
from torchvision import transforms
import random


# Later, apply augmentations when loading data, for example:


def drop_path(x, drop_prob: float = 0.1, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training: # if the drop probability =0 , or when the model is not training -> 
                                        # no dropout , just return the original
                                        # because : When inference , we just keep things the way is it, no dropout , or else it 
                                        # will ruin the result
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
                                                 # shape = (x.shape[0],) -> "shape" is now a 1D array,
                                                 #                       -> copy the first element of the shape of input x
                                                 # shape = + (1,) *(x.ndim-1) -> adding 3 more dimension (because x is 4-dimensions -> x.ndim-1 = 3)
                                                 #                            -> but in each dimension, only 1 element exists (in order to broadcase)
                                                 #  Example : X.shape -> (4,3,5,5) (4 samples , 3 channels, 5 width , 5 height)
                                                 # ->           shape -> (4,1,1,1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob) # this random_tensor have the above shape , and the value is either 0 or 1
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob) # in order to preserve the expected value of the output 
                                      # we dropped things, but we want the final expected value to be the same
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

### torch version too old for timm
### https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)




def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2 # THIS PADDING IS DOPED!!!!! It preserve the spatial dimension after doing
                                                                      # convolution with this specific dilation rate
                                                                      # This is why we don't need to be afraid about incompatible shape
                                                                      # when we concat channel-wise diffrent branches (diffrent dilation rate) 
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
            )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    


# This is the part where we fix the temporal module:
"""
class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=False,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0
                ),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation
                ),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)
        # print(len(self.branches))
        # initialize
        self.apply(weights_init)



    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out
"""
# This is the fixed MultiScale_TemporalConv
class DynamicGroupTCN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2], # we still use these dilations, but we are going to use dilation 2 and 1 multiple times 
                 residual=True,
                 residual_kernel_size=1,
                 drop_prob=0.3):

        super(DynamicGroupTCN, self).__init__()
        assert out_channels % (len(dilations) + 2) == 0

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2  

        # The "+2" in the line is because the architecture includes two additional branches apart from the ones created by dilations. 
        # These two branches are a 1x1 convolution and a 3x1 max pooling layer.
        branch_channels = out_channels // self.num_branches # this already divided the channels into branches (like our group)
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations) # we dont use this anymore, because now we decide to have 6 branches in total:
        else:
            kernel_size = [kernel_size]*len(dilations) # we dont use this anymore, because now we decide to have 6 branches in total:


        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0
                ),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(drop_prob),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation
                ),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])


        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)  
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        ))
        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        # print(len(self.branches))
        # initialize
        self.apply(weights_init)

    # fix this forward function so that it can process 2 param too
    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for branch in self.branches:
            out = branch(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out
    """
    def forward(self, joint_features, skeleton_features=None):
        

        #joint_features = joint_features.permute(0, 2, 1, 3)
        res_joint = self.residual(joint_features)
        branch_outs_joint = []
        for branch in self.branches:
            out_joint = branch(joint_features)
            branch_outs_joint.append(out_joint)
            out_joint = torch.cat(branch_outs_joint, dim=1)
            out_joint += res_joint

        # If skeleton_features is None, then only process joint_features
        if skeleton_features is None:
            return out_joint

        # If skeleton_features is provided, process it as well
        skeleton_features = skeleton_features.permute(0, 2, 1, 3)
        res_skeleton = self.residual(skeleton_features)
        branch_outs_skeleton = []
        for branch in self.branches:
            out_skeleton = branch(skeleton_features)
            branch_outs_skeleton.append(out_skeleton)

        out_skeleton = torch.cat(branch_outs_skeleton, dim=1)
        out_skeleton += res_skeleton

        # Concatenate the output
        out = torch.cat((out_joint, out_skeleton), dim=1)
        out = out.permute(0,2,1,3)
        return out
"""
# End of "This is the fixed MultiScale_TemporalConv"


# This is the additional D-JSF module:
"""
class DJSF(nn.Module):
    def __init__(self, num_joints, num_channels, num_groups):
        super(DJSF, self).__init__()

        self.num_joints = num_joints
        self.num_channels = num_channels
        self.num_groups = num_groups
        # X la N , C , T , V
        # Joint-level feature processing
        self.joint_pooling = nn.AvgPool1d(num_joints) # N,C,T
        self.joint_conv = nn.Conv1d(num_channels, num_channels, kernel_size=1) # N,C,T
        
        # Skeleton-level feature processing
        self.skeleton_conv = nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=1)
        
        # Dynamic joint-skeleton fusion
        self.dg_tcn_list = nn.ModuleList()
        # for i in range(num_groups):
        dg_tcn = DynamicGroupTCN(in_channels=num_channels,
                                     out_channels=num_channels,
                                     dilations=[1,2,3,4,5,6])  ### could be choosing lower dilation rate
        #self.dg_tcn_list.append(dg_tcn)
        
    def forward(self, x):
      print("the shape of x is : " + str(x.shape))
      # Joint-level feature processing
      joint_features = x.view(-1, self.num_channels, x.size(2))
      joint_features = self.joint_pooling(joint_features).squeeze(dim=-1)  # shape: (batch_size*num_groups, num_channels)
      # joint_features = joint_features.view(-1, self.num_groups, self.num_channels)
      # Skeleton-level feature processing
      x = x.mean(dim=3)  # Average out the joint dimension , we need to do this to turn skeleton_features 4D->3D
      skeleton_features = self.skeleton_conv(x)  # shape: (batch_size*num_groups, num_channels, seq_len)
      print(skeleton_features.shape)
    
      # Dynamic joint-skeleton fusion
      fused_features_list = []
      for i in range(self.num_groups):
          dg_tcn = self.dg_tcn_list[i]
          print("joint_features:", joint_features)
          fused_features = dg_tcn(joint_features,skeleton_features)
          fused_features_list.append(fused_features)
    
      # Concatenate and process fused features
      fused_features = torch.cat(fused_features_list, dim=-1)  # shape: (batch_size*num_groups, num_channels*num_groups)
      fused_features = fused_features.view(-1, self.num_groups, self.num_channels)  # shape: (batch_size, num_groups, num_channels)
    
      output = self.joint_conv(fused_features.permute(0, 2, 1))
      # output = self.joint_conv(fused_features)
      return output
"""
class DJSF(nn.Module):
    def __init__(self, num_joints, num_channels, dilations=[5,6]):
        super(DJSF, self).__init__()
        self.num_joints = num_joints
        self.num_channels = num_channels
        self.num_groups = len(dilations)+2
        
        # Define the modules used in the fusion process
        self.avg_pool = nn.AvgPool1d(kernel_size=num_joints)
        #self.tcn_skeleton = nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=1, groups=num_groups)
        self.tcn_skeleton = DynamicGroupTCN(in_channels=num_channels,
                                            out_channels=num_channels,
                                            dilations=dilations)
        #self.tcn_joint = nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=1)
        self.tcn_joint = DynamicGroupTCN(in_channels=num_channels,
                                            out_channels=num_channels,
                                            dilations=dilations)
        self.fc_gamma = nn.Linear(216*64*1, num_joints)
        
    def forward(self, x):
        x_joint = x
        # Perform average pooling to obtain skeleton-level features
        # x_skeleton = self.avg_pool(x_joint)
        x_skeleton=torch.mean(x_joint, dim=3)
        
        
        # Process skeleton-level and joint-level features with TCNs in parallel
        # x_skeleton_tcn = self.tcn_skeleton(x_skeleton.transpose(1, 2)).transpose(1, 2)
        # x_joint_tcn = self.tcn_joint(x_joint.transpose(1, 2)).transpose(1, 2)
        x_skeleton_tcn = self.tcn_skeleton(x_skeleton.transpose(0,1).transpose(1,2).unsqueeze(-1).transpose(1,2).transpose(0,1))
        x_joint_tcn = self.tcn_joint(x_joint.transpose(0,1).transpose(1,2).transpose(2,3).transpose(1,2).transpose(0,1))
        #print("shape of x_ske is: " + str(x_skeleton_tcn.shape))
        #print("shape of x_joint is: " + str(x_joint_tcn.shape))
        
        
        
        # Apply dynamic joint-skeleton fusion to merge skeleton-level and joint-level features
        gamma = torch.sigmoid(self.fc_gamma(x_skeleton_tcn.flatten(start_dim=1)))
        #print("shape of gamma is: " + str(gamma.shape))
        # Reshape gamma from [128, 25] to [128, 1, 1, 25]
        gamma = gamma.unsqueeze(1).unsqueeze(2)

        # Now, gamma has shape [128, 1, 1, 25] and x_skeleton_tcn has shape [128, 256, 64, 1]

        # Expand x and gamma so they have compatible shapes
        x_skeleton_tcn_expanded = x_skeleton_tcn.expand(-1, -1, -1, gamma.size(-1))
        gamma_expanded = gamma.expand(-1,x_skeleton_tcn.size(1), x_skeleton_tcn.size(2), -1)

        
        x_joint_fused = x_joint_tcn.transpose(0,3) + gamma_expanded* x_skeleton_tcn_expanded
        #x_joint_fused = x.transpose(0,1)
        return x_joint_fused


# End of "This is the additional D-JSF module"





#Overall, the unit_tcn module represents a basic building block of a TCN, consisting of a convolutional layer, batch normalization, and ReLU activation.
class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1), groups=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class MHSA(nn.Module):
    # A is adjacentcy matrix, and is hardcoded in the graph folder
    def __init__(self, dim_in, dim, A, num_heads=6, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1, insert_cls_layer=0, pe=False, num_point=25,
                 outer=True, layer=0,
                 **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.num_point = num_point
        self.layer = layer



        h1 = A.sum(0)
        h1[h1 != 0] = 1
        h = [None for _ in range(num_point)]
        h[0] = np.eye(num_point)
        h[1] = h1
        self.hops = 0*h[0]
        for i in range(2, num_point):
            h[i] = h[i-1] @ h1.transpose(0, 1)
            h[i][h[i] != 0] = 1

        for i in range(num_point-1, 0, -1):
            if np.any(h[i]-h[i-1]):
                h[i] = h[i] - h[i - 1]
                self.hops += i*h[i]
            else:
                continue

        self.hops = torch.tensor(self.hops).long()
        #
        self.rpe = nn.Parameter(torch.zeros((self.hops.max()+1, dim)))

        self.w1 = nn.Parameter(torch.zeros(num_heads, head_dim))



        A = A.sum(0)
        A[:, :] = 0

        self.outer = nn.Parameter(torch.stack([torch.eye(A.shape[-1]) for _ in range(num_heads)], dim=0), requires_grad=True)

        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.kv = nn.Conv2d(dim_in, dim * 2, 1, bias=qkv_bias)
        self.q = nn.Conv2d(dim_in, dim, 1, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)


        self.proj = nn.Conv2d(dim, dim, 1, groups=6)

        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)
        self.insert_cls_layer = insert_cls_layer

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, e):
        N, C, T, V = x.shape
        kv = self.kv(x).reshape(N, 2, self.num_heads, self.dim // self.num_heads, T, V).permute(1, 0, 4, 2, 5, 3)
        k, v = kv[0], kv[1]

        ## n t h v c
        q = self.q(x).reshape(N, self.num_heads, self.dim // self.num_heads, T, V).permute(0, 3, 1, 4, 2)

        e_k = e.reshape(N, self.num_heads, self.dim // self.num_heads, T, V).permute(0, 3, 1, 4, 2)
        #
        #
        pos_emb = self.rpe[self.hops]
        #
        k_r = pos_emb.view(V, V, self.num_heads, self.dim // self.num_heads)
        #
        b = torch.einsum("bthnc, nmhc->bthnm", q, k_r)
        #
        c = torch.einsum("bthnc, bthmc->bthnm", q, e_k)
        d = torch.einsum("hc, bthmc->bthm", self.w1, e_k).unsqueeze(-2)


        a = q @ k.transpose(-2, -1)

        attn = a + b + c + d


        attn = attn * self.scale

        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)


        x = (self.alpha * attn + self.outer) @ v
        # x = attn @ v


        x = x.transpose(3, 4).reshape(N, T, -1, V).transpose(1, 2)
        x = self.proj(x)

        x = self.proj_drop(x)
        return x

# using conv2d implementation after dimension permutation
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 num_heads=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x = self.fc1(x.transpose(1,2)).transpose(1,2)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        # x = self.fc2(x.transpose(1,2)).transpose(1,2)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class unit_vit(nn.Module):
    def __init__(self, dim_in, dim, A, num_of_heads, add_skip_connection=True,  qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer=0,
                insert_cls_layer=0, pe=False, num_point=25, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim_in)
        self.dim_in = dim_in
        self.dim = dim
        self.add_skip_connection = add_skip_connection
        self.num_point = num_point
        self.attn = MHSA(dim_in, dim, A, num_heads=num_of_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                             proj_drop=drop, insert_cls_layer=insert_cls_layer, pe=pe, num_point=num_point, layer=layer, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.dim_in != self.dim:
            self.skip_proj = nn.Conv2d(dim_in, dim, (1, 1), padding=(0, 0), bias=False)
        self.pe_proj = nn.Conv2d(dim_in, dim, 1, bias=False)
        self.pe = pe

    def forward(self, x, joint_label, groups):
        ## more efficient implementation
        label = F.one_hot(torch.tensor(joint_label)).float().to(x.device)
        z = x @ (label / label.sum(dim=0, keepdim=True))

        # w/o proj
        # z = z.permute(3, 0, 1, 2)
        # w/ proj
        z = self.pe_proj(z).permute(3, 0, 1, 2)

        e = z[joint_label].permute(1, 2, 3, 0)

        if self.add_skip_connection:
            if self.dim_in != self.dim:
                x = self.skip_proj(x) + self.drop_path(self.attn(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), e))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), e))
        else:
            x = self.drop_path(self.attn(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), e))

        # x = x + self.drop_path(self.mlp(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))

        return x
""" We need to comment this, and replace with the new TCN_Vit_unit
class TCN_ViT_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, num_of_heads=6, residual=True, kernel_size=5, dilations=[1,2], pe=False, num_point=25, layer=0):
        super(TCN_ViT_unit, self).__init__()
        self.vit1 = unit_vit(in_channels, out_channels, A, add_skip_connection=residual, num_of_heads=num_of_heads, pe=pe, num_point=num_point, layer=layer)
        # self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            dilations=dilations,
                                            # redisual=True has worse performance in the end
                                            residual=False)
        self.act = nn.ReLU(inplace=True)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, joint_label, groups):
        y = self.act(self.tcn1(self.vit1(x, joint_label, groups)) + self.residual(x))
        return y
"""
class TCN_ViT_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, num_of_heads=6, residual=True, kernel_size=5, dilations=[1,2], pe=False, num_point=25, layer=0):
        super(TCN_ViT_unit, self).__init__()
        self.vit1 = unit_vit(dim_in=in_channels,
                             dim=out_channels,
                             A=A,
                             add_skip_connection=residual,
                             num_of_heads=num_of_heads,
                             pe=pe,
                             num_point=num_point,
                             layer=layer)
        self.dg_tcn = DynamicGroupTCN(in_channels=out_channels, # fixed this for the mismatch size problem
                                      out_channels=out_channels,
                                      kernel_size=kernel_size)
        self.djsf = DJSF(num_joints=25,num_channels=out_channels)


        self.act = nn.ReLU(inplace=True)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        if not residual:
          self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
          self.residual = lambda x: x

        else:
          self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x,joint_label,groups):
        x = self.vit1(x,joint_label,groups)
        x = self.dg_tcn(x)
        x = self.djsf(x)
        # x = x+self.residual(x)
        x = self.act(x)
        return x

class Model(nn.Module):
    def __init__(self, num_class=60, num_point=20, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0.1, num_of_heads=9, joint_label=[]):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,25,25

        self.num_of_heads = num_of_heads
        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.joint_label = joint_label

        self.l1 = TCN_ViT_unit(3, 24*num_of_heads, A, residual=True, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=1)
        # * num_heads, effect of concatenation following the official implementation
        self.l2 = TCN_ViT_unit(24*num_of_heads, 24*num_of_heads, A, residual=True, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=2)
        self.l3 = TCN_ViT_unit(24*num_of_heads, 24*num_of_heads, A, residual=True, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=3)
        self.l4 = TCN_ViT_unit(24*num_of_heads, 24*num_of_heads, A, residual=True, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=4)
        self.l5 = TCN_ViT_unit(24*num_of_heads, 24*num_of_heads, A, residual=True, stride=2, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=5)
        self.l6 = TCN_ViT_unit(24*num_of_heads, 24*num_of_heads, A, residual=True, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=6)
        self.l7 = TCN_ViT_unit(24*num_of_heads, 24*num_of_heads, A, residual=True, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=7)
        # self.l8 = TCN_ViT_unit(24 * num_of_heads, 24 * num_of_heads, A, num_of_heads=num_of_heads)
        self.l8 = TCN_ViT_unit(24*num_of_heads, 24*num_of_heads, A, residual=True, stride=2, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=8)
        self.l9 = TCN_ViT_unit(24*num_of_heads, 24*num_of_heads, A, residual=True, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=9)
        # self.l10 = TCN_ViT_unit(24*num_of_heads, 24*num_of_heads, A, residual=True, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=10)
        # standard ce loss
        self.fc = nn.Linear(24*num_of_heads, num_class)
        
        # ## larger model
        # self.l1 = TCN_ViT_unit(3, 36 * num_of_heads, A, residual=True, num_of_heads=num_of_heads, pe=True,
        #                        num_point=num_point, layer=1)
        # # * num_heads, effect of concatenation following the official implementation
        # self.l2 = TCN_ViT_unit(36 * num_of_heads, 36 * num_of_heads, A, residual=True, num_of_heads=num_of_heads,
        #                        pe=True, num_point=num_point, layer=2)
        # self.l3 = TCN_ViT_unit(36 * num_of_heads, 36 * num_of_heads, A, residual=True, num_of_heads=num_of_heads,
        #                        pe=True, num_point=num_point, layer=3)
        # self.l4 = TCN_ViT_unit(36 * num_of_heads, 36 * num_of_heads, A, residual=True, num_of_heads=num_of_heads,
        #                        pe=True, num_point=num_point, layer=4)
        # self.l5 = TCN_ViT_unit(36 * num_of_heads, 36 * num_of_heads, A, residual=True, stride=2,
        #                        num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=5)
        # self.l6 = TCN_ViT_unit(36 * num_of_heads, 36 * num_of_heads, A, residual=True, num_of_heads=num_of_heads,
        #                        pe=True, num_point=num_point, layer=6)
        # self.l7 = TCN_ViT_unit(36 * num_of_heads, 36 * num_of_heads, A, residual=True, num_of_heads=num_of_heads,
        #                        pe=True, num_point=num_point, layer=7)
        # # self.l8 = TCN_ViT_unit(36 * num_of_heads, 36 * num_of_heads, A, num_of_heads=num_of_heads)
        # self.l8 = TCN_ViT_unit(36 * num_of_heads, 36 * num_of_heads, A, residual=True, stride=2,
        #                        num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=8)
        # self.l9 = TCN_ViT_unit(36 * num_of_heads, 36 * num_of_heads, A, residual=True, num_of_heads=num_of_heads,
        #                        pe=True, num_point=num_point, layer=9)
        # self.l10 = TCN_ViT_unit(36 * num_of_heads, 36 * num_of_heads, A, residual=True, num_of_heads=num_of_heads,
        #                         pe=True, num_point=num_point, layer=10)
        # # standard ce loss
        # self.fc = nn.Linear(36 * num_of_heads, num_class)

        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x, y):
        groups = []
        for num in range(max(self.joint_label)+1):
            groups.append([ind for ind, element in enumerate(self.joint_label) if element==num])
        
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)

        ## n, c, t, v
        x = x.view(N, M, V, C, T).contiguous().view(N * M, V, C, T).permute(0, 2, 3, 1)

        x = self.l1(x, self.joint_label, groups)
        x = self.l2(x, self.joint_label, groups)
        x = self.l3(x, self.joint_label, groups)
        x = self.l4(x, self.joint_label, groups)
        x = self.l5(x, self.joint_label, groups)
        x = self.l6(x, self.joint_label, groups)
        x = self.l7(x, self.joint_label, groups)
        x = self.l8(x, self.joint_label, groups)
        x = self.l9(x, self.joint_label, groups)
        # x = self.l10(x, self.joint_label, groups)

        # N*M, C, T, V
        _ , C, T, V = x.size()
        # spatial temporal average pooling
        x = x.reshape(N, M, C, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)
        x = self.fc(x)

        return x, y
