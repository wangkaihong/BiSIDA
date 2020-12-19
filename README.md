# BiSIDA: Bidirectional Style-induced Domain Adaptation
Pytorch implementation for "Consistency Regularization with High-dimensional Non-adversarial Source-guided Perturbation for Unsupervised Domain Adaptation in Segmentation" accepted by AAAI-2021.

Our bidirectional style-induced domain adaptation employs consistency regularization to efficiently exploit information from the unlabeled target domain dataset, requiring only a simple ****neural style transfer**** model. 

BiSIDA aligns domains by:
* transferring source images into the style of target images for supervised learning; 
* transferring target images into the style of source images to perform high-dimensional perturbation on the unlabeled target images for unsupervised learning. 

![Image of BiSIDA](https://github.com/wangkaihong/BiSIDA/blob/master/demo_img/pipeline.png)

 An example of our BiSIDA on the SYNTHIA-to-CityScapes benchmark eperiment. 

 ![Image of Source](https://github.com/wangkaihong/BiSIDA/blob/master/demo_img/vis.png)

# Usage

1. Download the pretrained VGG model required by both the our style transfer network and FCN, and put it into saved_models/.

   VGG initializations is available through this [link.](https://drive.google.com/file/d/11PbJLLd9C3-Aj4yiRbJoDgEZyfZn3dIv/view?usp=sharing)
   
2. Pretraining of our continuous style-induced image generator ([AdaIN](https://github.com/xunhuang1995/AdaIN-style)).

   > python adain/train/train_0_1.py
   
   An example of our continuous style-induced image generator transferring an image in SYNTHIA to a image in CityScapes with different alpha ranging from 0 to 1 with an increment of 0.2.
   
   ![Image of alpha](https://github.com/wangkaihong/BiSIDA/blob/master/demo_img/alpha.png)

   Note: Pretrained style transfer network is available through this  [link](https://drive.google.com/file/d/1lgoRj-M9c9kTKPPnmm2G5kdGY4K7G3-1/view?usp=sharing) and  should be placed in saved_models/.

3. Experiment on SYNTHIA-to-CityScapes benckmark

   > python train/train_synthia_vgg/train_synthia_vgg_experiment.py

4. Experiment on GTAV-to-CityScapes benckmark

   > python train/train_gta_vgg/train_gta_vgg_experiment.py
             
   
**Acknowledgment**

Code adapted from [BDL](https://github.com/liyunsheng13/BDL), [self ensemble visual domain adapt](https://github.com/wangkaihong/self-ensemble-visual-domain-adapt), and [fcn](https://github.com/wkentaro/fcn/). 
