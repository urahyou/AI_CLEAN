import numpy as np
import h5py
from torch.utils.data import Dataset
# import ehtim.const_def as ehc
from torchvision import transforms
from skimage import color

data_root = '/mnt/storage-ssd/zhouxinghui/VIC-DDPM/data'

def create_gaussian_mask(size, sigma=None, center=None):
    """
    创建一个高斯权重矩阵。
    
    参数:
    size (int): 矩阵的大小，生成 size x size 的矩阵。
    sigma (float): 高斯分布的标准差。如果为 None，则默认为 size/6。
    center (tuple): 高斯分布的中心坐标 (x, y)。如果为 None，则默认为矩阵的中心。
    
    返回:
    np.ndarray: 一个浮点型矩阵，表示高斯权重。
    """
    if sigma is None:
        sigma = size / 6  # 默认标准差为 size/6，可以根据需要调整
    if center is None:
        center = (size / 2, size / 2)
    
    Y, X = np.ogrid[:size, :size]
    dist_from_center = (X - center[0])**2 + (Y - center[1])**2
    gaussian_mask = np.exp(-dist_from_center / (2 * sigma**2))
    
    return gaussian_mask

def to_img(s_vis_real, s_vis_imag, uv_dense, npixels=128, domask=True, vis_noise_level=0):
    print(s_vis_imag.shape, s_vis_real.shape)
    if npixels != s_vis_real.shape[0]:
        print('尺寸不对！')
        return 
    nF = npixels
    
    # print('不去除边界效应 1')
    s_vis_imag[0,0] = 0        # 把虚部至0
    s_vis_imag[0,nF//2] = 0
    s_vis_imag[nF//2,0] = 0
    s_vis_imag[nF//2,nF//2] = 0
    
    if vis_noise_level != 0:
        vis_noise = np.random.normal(0, vis_noise_level, s_vis_real.shape)
        s_vis_real = s_vis_real + vis_noise
        s_vis_imag = s_vis_imag + vis_noise
    
    # 变成复数
    s_fft =  s_vis_real + 1j*s_vis_imag
    
     # NEW: set border to zero to counteract weird border issues
    s_fft[0,:] = 0.0
    s_fft[:,0] = 0.0
    s_fft[:,-1] = 0.0
    s_fft[-1,:] = 0.0
    
    # 加高斯锥函数
    if domask == True:
        mask = create_gaussian_mask(npixels, sigma=npixels/4)  # 创建高斯掩膜
        s_fft = s_fft * mask  # 应用掩膜，高斯加锥
    # mask = create_circular_mask(npixels, radius=npixels/2)  # 创建圆形掩膜
    # s_fft[~mask] = 0.0  # 应用掩膜，将圆外的值置为0 
    
    eht_fov  = 1.4108078120287498e-09 
    max_base = 8368481300.0
    # img_res = self.hparams.input_size 
    # 图像分辨率，npix
    img_res = npixels
    scale_ux= max_base * eht_fov/ img_res

    uv_dense_per=uv_dense
    u_dense, v_dense= np.unique(uv_dense_per[:,0]), np.unique(uv_dense_per[:,1]) # 这里就变成了128*1, 128*1
    # u_dense= np.linspace( u_dense.min(), u_dense.max(), len(u_dense)//2 * 2 ) # 128,
    # v_dense= np.linspace( v_dense.min(), v_dense.max(), len(v_dense)//2 * 2 ) # 128
    # 重新生成uv坐标
    u_dense = np.linspace(u_dense.min(), u_dense.max(), npixels) 
    v_dense = np.linspace(v_dense.min(), v_dense.max(), npixels)
    uv_arr= np.concatenate([np.expand_dims(u_dense,-1), np.expand_dims(v_dense,-1)], -1) # (128, 2)
    
    uv_arr= ((uv_arr+.5) * 2 -1.) * scale_ux # scaled input
    # np.expand_dims多塞一个纬, 所以s_fft 变成了(1, 128, 128)
    # _, img_recon, _ = make_dirtyim(uv_arr, np.expand_dims(s_fft,0), img_res, eht_fov, return_im=True)
    # print(img_recon)
    img_recon = make_im_np(uv_arr, np.expand_dims(s_fft,0), img_res, eht_fov, norm_fact=0.0087777, return_im=True)
    img_real = img_recon.real.squeeze(0)  # 删除第0纬
    img_imag = img_recon.imag.squeeze(0)
    img_recon = np.concatenate([img_real, img_imag], axis=0)
    img_recon = img_recon.reshape(2, npixels, npixels)
    return img_recon



# 这个加载的是eht_grid_128FC_200im_Galaxy10_DECals_full.h5
def load_h5_uvvis(fpath):
    print('--loading h5 file for eht sparse and dense {u,v,vis_re,vis_im} dataset...')
    with h5py.File(fpath, 'r') as F:
        u_sparse = np.array(F['u_sparse'])
        # u_sparse = F['u_sparse']
        v_sparse = np.array(F['v_sparse'])
        # v_sparse = F['v_sparse']
        vis_re_sparse = np.array(F['vis_re_sparse'])
        # vis_re_sparse = F['vis_re_sparse']
        vis_im_sparse = np.array(F['vis_im_sparse'])
        # vis_im_sparse = F['vis_im_sparse']
        u_dense = np.array(F['u_dense'])
        # u_dense = np.array(F['u_dense'])
        v_dense = np.array(F['v_dense'])
        # v_dense = np.array(F['v_dense'])
        vis_re_dense = np.array(F['vis_re_dense'])
        # vis_re_dense = F['vis_re_dense']
        vis_im_dense = np.array(F['vis_im_dense'])
        # vis_im_dense = F['vis_im_dense']
    print('Done--')
    return u_sparse, v_sparse, vis_re_sparse, vis_im_sparse, u_dense, v_dense, vis_re_dense, vis_im_dense


# 这个加载的是eht_cont_200im_Galaxy10_DECals_full.h5
def load_h5_uvvis_cont(fpath):
    print('--loading h5 file for eht continuous {u,v,vis_re,vis_im} dataset...')
    with h5py.File(fpath, 'r') as F:
        u_cont = np.array(F['u_cont'])
        # u_cont = F['u_cont'][:]
        v_cont = np.array(F['v_cont'])
        # v_cont = F['v_cont'][:]
        vis_re_cont = np.array(F['vis_re_cont'])
        # vis_re_cont = F['vis_re_cont'][:]
        vis_im_cont = np.array(F['vis_im_cont'])
        # vis_im_cont = F['vis_im_cont'][:]
    print('Done--')
    return u_cont, v_cont, vis_re_cont, vis_im_cont

    
# 使用DFT进行观测，假设visibility落在规则格子内, 返回numpy
def make_im_np(
    uv_arr, vis_arr, npix, fov, weighting='uniform', norm_fact=None, return_im=False, seperable_FFT=True, rescaled_pix=True):
    """Make the observation image using direct Fourier transform. 
    Assume the visibilities are on regulars grid in the continuous domain

        Args:
            uv_arr- U x 2 (U==V)
            vis_arr- B x U x V
            npix (int): The pixel size of the square output image.
            fov (float): The field of view of the square output image in radians.
            pulse (function): The function convolved with the pixel values for continuous image.
            weighting (str): 'uniform' or 'natural'
        Returns:
            (Image): an Image object with dirty image.
    """
    import math
    #print('you are using make_im_np...')
    if rescaled_pix:
        pdim = 1. #scaled input  # 射电里的pixel size
    else:
        pdim = fov / npix

    u = uv_arr[:,0]  # u的坐标
    v = uv_arr[:,1]  # v的坐标

    B, U, V= vis_arr.shape[0], vis_arr.shape[1], vis_arr.shape[2]
    #print(f'B:{B}, U:{U}, V:{V}') 
    # B:1, U:128, V:128
    assert U==V

    #TODO: xlist as input to speed up
    #DONE: calculate the scale of u*x and v*x directly
    #DONE: scaled by normfac
    # xlist 是一个一维数组，用于表示输出图像中每个像素在视场（Field of View, FOV）中的位置
    # xlist 中的值代表输出图像中每个像素在视场中的实际位置。这个位置是相对于视场中心的坐标，通常用于计算信号在图像中的分布
    xlist = np.arange(0, -npix, -1) * pdim + (pdim * npix) / 2.0 - pdim / 2.0
    

    # #--Sequence 1D Inverse DFT--#
    # 一维版的fft，就是分开行和列来做
    if seperable_FFT:
        X_coord= xlist.reshape(1, npix, 1, 1, 1)
        Y_coord= xlist.reshape(1, 1, npix, 1, 1)
        U_coord= u.reshape(1,1,1, U,1)
        V_coord= v.reshape(1,1,1, 1,V)
        Vis= vis_arr.reshape(B, 1, 1, U, V)
        #the inner integration (over u) 
        U_X= U_coord*X_coord  # u和x的内积
        #print(f'U_X:{U_X.shape}')
        # inner_integral= torch.sum(Vis * torch.exp(-2.j* math.pi* U_X) , dim=-2,keepdim=True) #B X 1 1 V
        inner_integral= np.mean(Vis * np.exp(-2.j* math.pi* U_X) , axis=-2) #B X 1 1 V
        # print("inner_integral:",inner_integral.shape)
        inner_integral = np.expand_dims(inner_integral, 2)
        #the outer integration (over v) 
        V_Y= V_coord*Y_coord  # v和y的内积
        # outer_integral= torch.sum(inner_integral * torch.exp(-2.j*math.pi* V_Y), dim=-1, keepdim=True ) # B X Y 1 1
        # 这里计算外积
        outer_integral= np.mean(inner_integral * np.exp(-2.j*math.pi* V_Y), axis=-1 ) # B X Y 1 1
        # print("outer_integral:",outer_integral.shape)
        image_complex= outer_integral.squeeze(-1) # B X Y
    else:
        # 二维版的fft，一次性做完
        #--2D raw version IDFT--#
        X_coord= xlist.reshape(1, npix, 1, 1, 1).expand(B,npix,npix, U,V)
        Y_coord= xlist.reshape(1, 1, npix, 1, 1).expand_as(X_coord)
        U_coord= u.reshape(1,1,1, U,1).expand_as(X_coord)
        V_coord= v.reshape(1,1,1, 1,V).expand_as(X_coord)
        U_X= U_coord*X_coord
        V_Y= V_coord*Y_coord
        Vis= vis_arr.reshape(B, 1, 1, U, V).expand_as(X_coord)
        image_complex= np.mean(Vis * np.exp(-2.j*math.pi*(U_X + V_Y)), axis=-1).mean(axis=-1)

    if norm_fact is not None:
        image_complex= image_complex* norm_fact

    
    # import pdb; pdb.set_trace()
    return image_complex

