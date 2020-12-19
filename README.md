# BiSIDA: Bidirectional Style-induced Domain Adaptation
Pytorch implementation for "Consistency Regularization with High-dimensional Non-adversarial Source-guided Perturbation for Unsupervised Domain Adaptation in Segmentation" accepted by AAAI-2021.

Our bidirectional style-induced domain adaptation employs consistency regularization to efficiently exploit information from the unlabeled target domain dataset, requiring only a simple ****neural style transfer**** model. 

BiSIDA aligns domains by:
* transferring source images into the style of target images to perform supervised learning; 
* transferring target images into the style of source images to perform high-dimensional perturbation on the unlabeled target images through unsupervised learning. 

![Image of BiSIDA](https://github.com/wangkaihong/BiSIDA/blob/master/demo_img/pipeline.png)

 An example of our BiSIDA from SYNTHIA to CityScapes. 

 ![Image of Source](https://github.com/wangkaihong/BiSIDA/blob/master/demo_img/vis.png)

# Usage

1. Pretraining of the style transfer network

   > python adain/train/train_0_1.py

2. Experiment on SYNTHIA-to-CityScapes benckmark

   > python train/train_synthia_vgg/train_synthia_vgg_experiment.py

3. Experiment on GTAV-to-CityScapes benckmark

   > python train/train_gta_vgg/train_gta_vgg_experiment.py
       
4. Pretrained Models

   VGG initializations is available through this [link.](https://drive.google.com/file/d/11PbJLLd9C3-Aj4yiRbJoDgEZyfZn3dIv/view?usp=sharing)
   
   Pretrained style transfer network is available through this  [link.](https://drive.google.com/file/d/1lgoRj-M9c9kTKPPnmm2G5kdGY4K7G3-1/view?usp=sharing)
      
   
**Acknowledgment**

Code adapted from [BDL](https://github.com/liyunsheng13/BDL), [self ensemble visual domain adapt](https://github.com/wangkaihong/self-ensemble-visual-domain-adapt), and [fcn](https://github.com/wkentaro/fcn/). 
