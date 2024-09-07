from download import download
from mindspore.dataset import Cifar10Dataset
url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz"

path = download(url, "./", kind="tar.gz", replace=True)

dataset = Cifar10Dataset("./cifar-10-batches-bin/")