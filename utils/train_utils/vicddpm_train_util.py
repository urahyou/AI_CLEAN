from utils.train_utils.ddpm_train_util import *


# 这是一个高级封装，详细的要看父类DDPMTrainLoop
class VICDDPMTrainLoop(DDPMTrainLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def batch_process(self, batch):
        # modify condition so that it only contains the information we need.
        batch, cond = batch
        # cond应该就是VIC-DDPM里面的condition？
        # 这里只需要取出用到的训练数据，可以看得到训练只需要三个数据，一个是uv坐标，一个是脏图，另一个是稀疏的visibility
        cond = {
            k: cond[k] for k in ["uv_coords", "image_dir", "vis_sparse"]
        }
        return batch, cond
