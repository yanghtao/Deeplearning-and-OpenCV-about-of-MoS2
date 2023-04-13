# DL_OpenCV_MoS2
MoS2因其优异的光电性能在近年来被研究者们广泛关注，而对于低维MoS2，其独特的能带结构(单层MoS2具有直接带隙，多层则具有间接带隙)使得研究者们在做深入研究时需要对其厚度进行表征，而目前制备层状MoS2的方法大多为机械剥离和CVD等，这些方法在制备的MoS2都具有分布不均匀且厚度难确定的特点，研究者们在研究其性能时，进行表征的过程耗时耗力，因此，我们通过深度学习算法，来进行机器自动识别
厚度信息，以减轻研究者们的工作量，促进研究进程。此外，最近有研究者发现CVD制备的MoS2其因单双层的扭角不同，会使得其性能发生改变，因此，我们在语义分割网络模型识别得到厚度的基础上，通过OpenCV图像处理算法，对MoS2样品进行自动检测其厚度，面积和扭角等信息。


Recently, MoS2 has been widely studied due to its excellent optoelectronic properties. For low-dimensional MoS2, its unique band structure (single-layer MoS2 has a direct band gap, while multiple layers have an indirect band gap) requires researchers to characterize its thickness in-depth. Currently, most methods for preparing layered MoS2 are mechanical exfoliation and CVD, which produce MoS2 with uneven distribution and difficult-to-determine thickness. Therefore, researchers spend a lot of time and effort characterizing its properties. To alleviate their workload and promote research progress, we use deep learning algorithms to automatically recognize thickness information. In addition, recent studies have found that the performance of CVD-prepared MoS2 changes due to different twist angles between single and double layers. Therefore, based on the semantic segmentation network model's recognition of thickness, we use OpenCV image processing algorithms to automatically detect thickness, area, and twist angle information of MoS2 samples.


# Prerequisites
python == 3.8

pytorch == 1.10.1

opencv-python == 4.5.6

please refer addtional packages in ```requearst.txt``` file

# 目录结构描述
为了方便大家使用我们提供的代码去自己制作数据集进行训练，或复现我们的实验，这里我们将整个语义分割所需的全部流程都进行详细说明，在此，仅简单介绍各模块的功能，具体如何使用，可产靠各模块内部的ReadMe文件，且结合文章的附录部分。
```
  ├── 01_Crop_Image: 裁剪大尺寸图像为规格统一的指定大小图像
  ├── 02_Labelme_to_dataset: 经LabelMe标注后的json文件转换为数据集格式
  ├── 03_Voc_dataset: 基于VOC 2007的语义分割网络模型所需数据集存放格式
  ├── 04_Segmentation_code: 语义分割网络模型
  └── 05_OpenCV: 基于OpenCV的图像处理获取MoS2扭角的pyhton文件
```

# 使用方法
  我们提供了训练好的模型以供大家尝试。可以通过Google drive下载并按照附录或模块内部`ReadMe.md`尝试使用。
  link:https://drive.google.com/drive/folders/1xYEWDbPEMBjnJhE0Cp-JYJ0Mml94I9bW?usp=share_link
  如果想要训练自己的数据集，请结合附录即各模块内部ReadMe文件使用。
