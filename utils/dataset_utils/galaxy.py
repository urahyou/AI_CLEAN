from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms

from utils.galaxy_data_utils.transform_util import *

# import pytorch_lightning as pl


from torch.utils.data import Dataset
import h5py
import numpy as np
import matplotlib.pyplot as plt
from utils.dataset_utils.data_ehtim_cont import *
import torch

transform = transforms.Resize((128, 128))

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


# 从sparse visbility转变成图像
# s_vis_real实部
# s_vis_imag虚部
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


def to_img1(s_vis_real, s_vis_imag, uv_dense, npixels=128):
    # s_vis_real （128，128）visbility的实部
    # npixels = 256
    if npixels != s_vis_real.shape[0]:
        print('尺寸不对！')
        return 
    nF = npixels
    
    # print('不去除边界效应 1')
    s_vis_imag[0,0] = 0        # 把虚部至0
    s_vis_imag[0,nF//2] = 0
    s_vis_imag[nF//2,0] = 0
    s_vis_imag[nF//2,nF//2] = 0
    
    # 变成复数
    s_fft =  s_vis_real + 1j*s_vis_imag

    # print('不去除边界效应 1')
    # NEW: set border to zero to counteract weird border issues
    s_fft[0,:] = 0.0
    s_fft[:,0] = 0.0
    s_fft[:,-1] = 0.0
    s_fft[-1,:] = 0.0

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


# 这个是数据集的加载类
class GalaxyDataset(Dataset):
    '''
    EHT-imaged dataset (load precomputed)  # 加载是数据是预先经过计算的
    ''' 
    def __init__(self,  
            dset_name = 'Galaxy10', # 'MNIST'
            data_path = f'{data_root}/eht_grid_128FC_200im_Galaxy10_DECals_full.h5', 
            data_path_imgs = f'{data_root}/Galaxy10_DECals.h5', 
            data_path_cont = f'{data_root}/eht_cont_200im_Galaxy10_DECals_full.h5',
            img_res = 128,
            pre_normalize = False,
            vis_noise_level=0
            ):
        self.vis_noise_level = vis_noise_level
        # get spectral data
        # 这里那到的是grid visibility数据
        u_sparse, v_sparse, vis_re_sparse, vis_im_sparse, u_dense, v_dense, vis_re_dense, vis_im_dense = load_h5_uvvis(data_path)
        print(u_sparse)
        # print(u_sparse.shape, v_sparse.shape, vis_re_sparse.shape, vis_im_sparse.shape, u_dense.shape, v_dense.shape, vis_re_dense.shape, vis_im_dense.shape)
        self.mask = np.load(f"{data_root}/mask.npy")
        self.sort_indices = np.load(f"{data_root}/sort_indices.npy")
        # 这里把实部和虚部stack到一起
        uv_sparse = np.stack((u_sparse.flatten(), v_sparse.flatten()), axis=1)
        uv_dense = np.stack((u_dense.flatten(), v_dense.flatten()), axis=1)  # (16384,2)
        fourier_resolution = int(len(uv_dense)**(0.5)) # 算出来是128，其实也就是图像的像素大小
        self.fourier_res = fourier_resolution

        # rescale uv to (-0.5, 0.5)
        max_base = np.max(uv_sparse)
        # 稍微放缩一下，放缩到-0.5到0.5
        uv_dense_scaled = np.rint((uv_dense+max_base) / max_base * (fourier_resolution-1)/2) / (fourier_resolution-1) - 0.5
        self.uv_dense = uv_dense_scaled
        self.vis_re_dense = vis_re_dense  # 这个下面用到
        self.vis_im_dense = vis_im_dense  # 其实只有这两个是用到的
        # TODO: double check un-scaling if continuous (originally scaled with sparse) 
        # should be ok bc dataset generation was scaled to max baseline, so np.max(uv_sparse)=np.max(uv_cont)

        # use sparse continuous data
        # 默认就会使用这个
        if data_path_cont: # 如果eht_cont_200im_Galaxy10_DECals_full.h5存在就使用连续的visibility
            print('using sparse continuous visibility data..')
            u_cont, v_cont, vis_re_cont, vis_im_cont = load_h5_uvvis_cont(data_path_cont)
            uv_cont = np.stack((u_cont.flatten(), v_cont.flatten()), axis=1)
            uv_cont_scaled = np.rint((uv_cont+max_base) / max_base * (fourier_resolution-1)/2) / (fourier_resolution-1) - 0.5
            self.uv_sparse = uv_cont_scaled
            self.vis_re_sparse = vis_re_cont
            self.vis_im_sparse = vis_im_cont
            
        # use sparse grid data
        else:  # 使用griding之后的visibility
            print('using sparse grid visibility data..')
            # np.rint是根据四舍五入来取整
            uv_sparse_scaled = np.rint((uv_sparse+max_base) / max_base * (fourier_resolution-1)/2) / (fourier_resolution-1) - 0.5
            self.uv_sparse = uv_sparse_scaled
            self.vis_re_sparse = vis_re_sparse
            self.vis_im_sparse = vis_im_sparse
        
        # load GT images
        # 这个是图像的分辨率， 默认是128
        self.img_res = img_res 
        
        if dset_name == 'Galaxy10' or 'Galaxy10_DECals':
            if data_path_imgs:
                self.img_dataset = Galaxy10_Dataset(data_path_imgs, None)
            else:  # if loading img data is not necessary
                self.img_dataset = None

        else:
            print('[ MNIST | Galaxy10 | Galaxy10_DECals ]')
            raise NotImplementedError
            
        # pre-normalize data? (disable for phase loss)
        self.pre_normalize = pre_normalize
            

    def __getitem__(self, idx):
        # match data structure of TransEncoder
        # 这里把实部和虚部都stack起来了,idx是第几个
        # vis_dense的第0纬度是16384 == 128*128
        # vis_dense是满覆盖的uv的visibility，用来得到原图
        vis_dense = np.stack((self.vis_re_dense[:,idx], self.vis_im_dense[:,idx]), axis=1)

        # 实部
        vis_real = self.vis_re_sparse[:,idx].astype(np.float32)
        # 虚部
        vis_imag = self.vis_im_sparse[:,idx].astype(np.float32)
        if self.pre_normalize == True:
            padding = 50 ## TODO make this actual hyperparam
            real_min, real_max= np.amin(vis_real)-padding, np.amax(vis_real)+padding
            imag_min, imag_max= np.amin(vis_imag)-padding, np.amax(vis_imag)+padding
            vis_real_normed = (vis_real - real_min) / (real_max - real_min)
            vis_imag_normed = (vis_imag - imag_min) / (imag_max - imag_min)
            vis_sparse = np.stack([vis_real_normed, vis_imag_normed], axis=1) 
        else:
            # 非满覆盖的uv，用来成脏图
            vis_sparse = np.stack([vis_real, vis_imag], axis=1)
        
        
        # 对visibility的数据进行重新排序，其实好像也就是旋转了一下90度  
        #sorted_vis = vis_dense[self.sort_indices]  # (16384)->(128, 128) 
        sorted_vis = vis_dense
        # 这里进行了一个reshape，这个应该就是相当于把1维的数据转变成了2维的,这里之所以是2channel是因为有实部和虚部两个通道
        reshaped_vis = sorted_vis.reshape((128, 128, 2),order='C')
        
        vis_dense = reshaped_vis.transpose((2, 0, 1)) # （2，128，128）
       
        # 这个是mask之后的visibility数据，而且是已经grided之后的数据
        vis_zf = vis_dense * self.mask  # 

        # 使用visibility进行成像
        img = to_img(vis_dense[0], vis_dense[1], self.uv_dense, domask=True)  # 这个是成像的函数，传入的是实部和虚部
        # img = self.img_dataset[idx][0]  # 这个是原图,256x256
        # img = transform(img)  # 这个是把原图resize成128x128
        # img = np.array(img)  # 这个是把PIL转变成numpy
        # 所以这里传进行的数据都是已经grided之后的数据
        # 这里成的是脏图片, 这里的vis_zf是使用griding之后的mask对全采样的visibility进行mask
        image_dir = to_img(vis_zf[0], vis_zf[1], self.uv_dense, domask=False, vis_noise_level=self.vis_noise_level)  # 这个是成像的函数，传入的是实部和虚部
        # _,image_dir,_ = make_dirtyim(self.uv_dense, np.expand_dims(vis_zf,0), self.img_res, 1.4108078120287498e-09, return_im=True)
        image_dir[1] = image_dir[0] # 这里再复制一个是什么鬼
        scale_coeff = 1. / np.max(np.abs(image_dir))  # 一个放缩因子，
        # 这里下面的注释是我加的，
        image_dir = image_dir * scale_coeff  # 每张图片都处以最大值
        vis_zf = vis_zf * scale_coeff   # masked之后的visibility也要进行放缩
        vis_dense = vis_dense * scale_coeff # 这个是dense的，也就是还没有mask的visibility，但应该也是落在规则的格子里面的，不然上面的mask操作不太可能
        img = img * scale_coeff   # GT也要进行一下放缩
      
        img[1] = img[0]
        
        args_dict = {
            "image": img.astype(np.float32),
            "image_dir": image_dir.astype(np.float32),
            "vis": vis_dense.astype(np.float32),  
            "vis_zf": vis_zf.astype(np.float32),  
            "mask": self.mask.astype(np.float32),
            "mask_c": np.ones((2,128,128)).astype(np.float32),
            "scale_coeff": scale_coeff,
            "uv_coords": self.uv_sparse.astype(np.float32),  # 对应望远镜阵列的稀疏uv坐标，(1660, 2)
            "vis_sparse": vis_sparse.astype(np.float32),  # 对应离散的uv坐标的visbility的值，(1660, 2)实部和虚部
            "acquisition": "none",
            "file_name": "none",
            "slice_index": "none",
        }
        return img.astype(np.float32), args_dict


    def __len__(self):
        return len(self.vis_re_sparse[0,:])




