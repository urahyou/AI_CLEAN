from utils.dataset_utils import galaxy


# 训练数据集的默认参数
def training_dataset_defaults():
    """
    Defaults for training galaxy dataset.
    """
    return dict(
        dataset="galaxy",
        data_dir="../datasets/galaxy/knee_singlecoil_train",
        data_info_list_path="",
        batch_size=1,
        acceleration=4,
        random_flip=False,
        is_distributed=True,
        num_workers=0,
    )

# 创建数据集
def create_training_dataset(
        dataset,
        data_dir,
        data_info_list_path,
        batch_size,
        acceleration,
        random_flip,
        is_distributed,
        num_workers,
):
    # 数据加载器，是一个函数，可以yield数据
    load_data = galaxy.load_data
    return load_data(
        data_dir=data_dir,
        data_info_list_path=data_info_list_path,
        batch_size=batch_size,
        random_flip=random_flip,
        is_distributed=is_distributed,
        is_train=True,
        post_process=None,
        num_workers=num_workers,
    )


def test_dataset_defaults():
    """
    Defaults for test galaxy dataset.
    """
    return dict(
        dataset="galaxy",
        data_dir="../dataset/galaxy/knee_singlecoil_val",
        data_info_list_path="",
        batch_size=1,
        acceleration=4,
        random_flip=False,
        is_distributed=True,
        num_workers=0,
    )


def create_test_dataset(
        dataset,
        data_dir,
        data_info_list_path,
        batch_size,
        acceleration,
        random_flip,
        is_distributed,
        num_workers,
):

    load_data = galaxy.load_data
    return load_data(
        data_dir=data_dir,
        data_info_list_path=data_info_list_path,
        batch_size=batch_size,
        random_flip=random_flip,
        is_distributed=is_distributed,
        is_train=False,
        post_process=None,
        num_workers=num_workers,
    )
