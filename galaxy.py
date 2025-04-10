# ref: https://github.com/RapidsAtHKUST/VIC-DDPM
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler

# from utils.dataset_utils.data_ehtim_cont import ehc
import ehtim.const_def as ehc
# from utils.galaxy_data_utils.transform_util import *

import pytorch_lightning as pl


from torch.utils.data import Dataset
import h5py
import numpy as np
import matplotlib.pyplot as plt
# from utils.dataset_utils.data_ehtim_cont import *
# import torch
from skimage import transform
from scipy.signal import convolve2d
from scipy.optimize import curve_fit

def fit_clean_beam(psf):
    """
    从给定的二维PSF拟合出一个高斯形状的clean beam。

    参数:
    psf (numpy.ndarray): 二维的PSF数组。

    返回:
    clean_beam (numpy.ndarray): 拟合后的clean beam数组。
    fit_params (dict): 拟合参数，包括振幅、中心位置、标准差和旋转角度。
    """
    # 高斯函数模型
    def gaussian_2d(xy, amp, x0, y0, sigma_x, sigma_y, theta):
        x, y = xy
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = amp * np.exp( - (a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))
        return g.ravel()

    # 创建坐标网格
    x = np.linspace(0, psf.shape[1] - 1, psf.shape[1])
    y = np.linspace(0, psf.shape[0] - 1, psf.shape[0])
    x, y = np.meshgrid(x, y)

    # 将网格数据和PSF数据展平，用于拟合
    initial_guess = (psf.max(), psf.shape[1] / 2, psf.shape[0] / 2, 10, 10, 0)  # 初始猜测值
    popt, pcov = curve_fit(gaussian_2d, (x, y), psf.ravel(), p0=initial_guess)

    # 提取拟合参数
    amp, x0, y0, sigma_x, sigma_y, theta = popt
    fit_params = {
        'amplitude': amp,
        'center_x': x0,
        'center_y': y0,
        'sigma_x': sigma_x,
        'sigma_y': sigma_y,
        'theta': theta
    }

    # 使用拟合参数生成Clean Beam
    clean_beam = gaussian_2d((x, y), *popt).reshape(psf.shape)

    return clean_beam, fit_params


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def to_img(s_vis_real, s_vis_imag, uv_dense):
    nF = 128
    s_vis_imag[0,0] = 0
    s_vis_imag[0,nF//2] = 0
    s_vis_imag[nF//2,0] = 0
    s_vis_imag[nF//2,nF//2] = 0

    s_fft =  s_vis_real + 1j*s_vis_imag

    # NEW: set border to zero to counteract weird border issues
    s_fft[0,:] = 0.0
    s_fft[:,0] = 0.0
    s_fft[:,-1] = 0.0
    s_fft[-1,:] = 0.0

    eht_fov  = 1.4108078120287498e-09 
    max_base = 8368481300.0
    # img_res = self.hparams.input_size 
    img_res = 128
    scale_ux= max_base * eht_fov/ img_res


    uv_dense_per=uv_dense
    u_dense, v_dense= np.unique(uv_dense_per[:,0]), np.unique(uv_dense_per[:,1])
    u_dense= np.linspace( u_dense.min(), u_dense.max(), len(u_dense)//2 * 2 )
    v_dense= np.linspace( v_dense.min(), v_dense.max(), len(v_dense)//2 * 2 )
    uv_arr= np.concatenate([np.expand_dims(u_dense,-1), np.expand_dims(v_dense,-1)], -1)

    uv_arr= ((uv_arr+.5) * 2 -1.) * scale_ux # scaled input
    img_recon = make_im_np(uv_arr, np.expand_dims(s_fft,0), img_res, eht_fov, norm_fact=0.0087777, return_im=True)
    img_real = img_recon.real.squeeze(0)
    # img_imag = img_recon.imag.squeeze(0)
    # img_recon = np.concatenate([img_real, img_imag], axis=0)
    # img_recon = img_recon.reshape(2, 128, 128)
    return img_real


def load_h5_uvvis(fpath):
    print('--loading h5 file for eht sparse and dense {u,v,vis_re,vis_im} dataset...')
    with h5py.File(fpath, 'r') as F:
        u_sparse = np.array(F['u_sparse'])
        v_sparse = np.array(F['v_sparse'])
        vis_re_sparse = np.array(F['vis_re_sparse'])
        vis_im_sparse = np.array(F['vis_im_sparse'])
        u_dense = np.array(F['u_dense'])
        v_dense = np.array(F['v_dense'])
        vis_re_dense = np.array(F['vis_re_dense'])
        vis_im_dense = np.array(F['vis_im_dense'])
    print('Done--')
    return u_sparse, v_sparse, vis_re_sparse, vis_im_sparse, u_dense, v_dense, vis_re_dense, vis_im_dense



def load_h5_uvvis_cont(fpath):
    print('--loading h5 file for eht continuous {u,v,vis_re,vis_im} dataset...')
    with h5py.File(fpath, 'r') as F:
        u_cont = np.array(F['u_cont'])
        v_cont = np.array(F['v_cont'])
        vis_re_cont = np.array(F['vis_re_cont'])
        vis_im_cont = np.array(F['vis_im_cont'])
    print('Done--')
    return u_cont, v_cont, vis_re_cont, vis_im_cont


def load_h5(fpath):
    print('--loading h5 file for Galaxy10 dataset...')
    with h5py.File(fpath, 'r') as F:
        x = np.array(F['images'])
        y = F['ans']
    print('Done--')

    return x, y


data_root = '/home/zhouxinghui/git/VIC-DDPM-modified/data'

class GalaxyDataset(Dataset):
    '''
    EHT-imaged dataset (load precomputed)
    ''' 
    def __init__(self,  
            dset_name = 'Galaxy10', # 'MNIST'
            data_path = f'{data_root}/eht_grid_128FC_200im_Galaxy10_DECals_full.h5', 
            data_path_imgs = f'{data_root}/Galaxy10_DECals.h5', 
            img_res = 128,
            ):
        
        self.imgs, _ = load_h5(data_path_imgs)
        # get spectral data
        u_sparse, v_sparse, vis_re_sparse, vis_im_sparse, u_dense, v_dense, vis_re_dense, vis_im_dense = load_h5_uvvis(data_path)
        # print(u_sparse.shape, v_sparse.shape, vis_re_sparse.shape, vis_im_sparse.shape, u_dense.shape, v_dense.shape, vis_re_dense.shape, vis_im_dense.shape)
        self.mask = np.load("./data/mask.npy")
        self.sort_indices = np.load("./data/sort_indices.npy")
        uv_sparse = np.stack((u_sparse.flatten(), v_sparse.flatten()), axis=1)
        uv_dense = np.stack((u_dense.flatten(), v_dense.flatten()), axis=1)
        fourier_resolution = int(len(uv_dense)**(0.5))
        self.fourier_res = fourier_resolution

        # rescale uv to (-0.5, 0.5)
        max_base = np.max(uv_sparse)
        uv_dense_scaled = np.rint((uv_dense+max_base) / max_base * (fourier_resolution-1)/2) / (fourier_resolution-1) - 0.5
        self.uv_dense = uv_dense_scaled
        self.vis_re_dense = vis_re_dense
        self.vis_im_dense = vis_im_dense
        # TODO: double check un-scaling if continuous (originally scaled with sparse) 
        # should be ok bc dataset generation was scaled to max baseline, so np.max(uv_sparse)=np.max(uv_cont)
            
        # 计算psf和clean_beam
        self.psf = to_img(self.mask[0], self.mask[1], self.uv_dense)
        self.clean_beam , _ = fit_clean_beam(self.psf)
            
        print('using sparse grid visibility data..')
        uv_sparse_scaled = np.rint((uv_sparse+max_base) / max_base * (fourier_resolution-1)/2) / (fourier_resolution-1) - 0.5
        self.uv_sparse = uv_sparse_scaled
        self.vis_re_sparse = vis_re_sparse
        self.vis_im_sparse = vis_im_sparse
        
        # load GT images
        self.img_res = img_res 
        
            

    def __getitem__(self, idx):

#        # match data structure of TransEncoder
        vis_dense = np.stack((self.vis_re_dense[:,idx], self.vis_im_dense[:,idx]), axis=1)
        
        # normalize to -0.5,0.5

        vis_real = self.vis_re_sparse[:,idx].astype(np.float32)
        vis_imag = self.vis_im_sparse[:,idx].astype(np.float32)
       
        vis_sparse = np.stack([vis_real, vis_imag], axis=1)
        
        sorted_vis = vis_dense[self.sort_indices]
 
        reshaped_vis = sorted_vis.reshape((128, 128, 2),order='C')
        vis_dense = reshaped_vis.transpose((2, 0, 1))
        vis_zf = vis_dense * self.mask

        # img = to_img(vis_dense[0], vis_dense[1], self.uv_dense)
        image_dir = to_img(vis_zf[0], vis_zf[1], self.uv_dense)
        # clean_mask = np.ones_like(self.mask[0])
        # clean_beam = to_img(clean_mask, clean_mask, self.uv_dense)
        # image_dir[1] = image_dir[0]
        scale_coeff = 1. / np.max(np.abs(image_dir))
        image_dir = image_dir * scale_coeff
        vis_zf = vis_zf * scale_coeff 
        vis_dense = vis_dense * scale_coeff
       
        
        gt = self.imgs[idx]
        gt = rgb2gray(gt)
        gt = np.transpose(gt)
        gt = transform.resize(gt, (128, 128), anti_aliasing=True)
        target = convolve2d(gt, self.clean_beam, mode='same', boundary='wrap') 
        
        
        # target = target.transpose(2,0,1)

        args_dict = {
            "idx": idx,
            "target": target,
            "gt": gt,
            # "image": img.astype(np.float32),
            "image_dir": image_dir.astype(np.float32),
            "vis": vis_dense.astype(np.float32),  
            "vis_zf": vis_zf.astype(np.float32),  
            "mask": self.mask.astype(np.float32),
            "mask_c": np.ones((2,128,128)).astype(np.float32),
            "scale_coeff": scale_coeff,
            "uv_coords": self.uv_sparse.astype(np.float32),
            "vis_sparse": vis_sparse.astype(np.float32),
            "acquisition": "none",
            "file_name": "none",
            "slice_index": "none",
        }
        return args_dict

    def __len__(self):
        return len(self.vis_re_sparse[0,:])



# 使用DFT进行观测，假设visibility落在规则格子内, 返回numpy
def make_im_np(
    uv_arr, vis_arr, npix, fov, pulse=ehc.PULSE_DEFAULT, weighting='uniform', norm_fact=None, return_im=False, seperable_FFT=True, rescaled_pix=True):
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
