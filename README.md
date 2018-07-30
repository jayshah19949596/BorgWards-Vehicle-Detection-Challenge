## Challenge Description
----

### Introduction to Challenge
----
- This coding test does not necessarily reflect or indicate what we are working on at Borgward R&D Silicon Valley.-  - The test is just to provide a way for us to estimate our candidate’s coding skills
- We are not looking for the exact correct answer or solutions, we are more looking for the way how you solve this problem, the way how you code, code style, code formality, code efficiencies, and most importantly the innovative thinking & ideas that you express in your algorithms. 
- Please do NOT spend more than 2 hours on this test, and try your best to finish the test with most efficient and practical code.

### Problem
----
- BDD100K is a large-scale diverse driving video database: http://bair.berkeley.edu/blog/2018/05/30/bdd/. There are different tasks of perception algorithms such as object detection, semantic segmentations, lane detections that can be developed and tested using this dataset.
- Let's now focus on one task called semantic segmentation. Semantic segmentation is so called "dense" prediction problem, where predictions are demanded at per pixel level. 
- Segmentation algorithms usually involves deep neural network, with approaches such as "encoder-decoder" architectures, "dilated convolutional layers", "multi-scale receptive field", so on and so forth.
- We are not going to ask you to code a DNN model to perform semantic segmentation of course for an interview test. Instead, we are working some handy tools as shown below in the figure: we need some code or algorithms to draw a bounding box around "car" or "vehicle pixels", in the example figure below, we have two cars in the raw image on the left, and assuming a prefect semantic segmentation DNN can give you a semantic map as shown on the right, we want to detect the boundaries or bounding boxes for the two vehicles, white small vehicle on the left, and the red big SUV on the right.

[image1]: ./image_rsrcs/problem.png "Problem"
![Problem][image1]

- Of course, this tasks is getting more and more challenging when vehicles have a lot of overlap on the image and shape becomes difficult to distinguish between different vehicles, but to get started we can assume the overlap is not huge.
- You can use any programming language that you are comfortable with (c++, python preferred), and we'd prefer you code the algorithm yourself instead of calling a library such as OpenCV. 
- If you need some small sample data to test the code, we have a few images here from BDD100K: https://borgward.atlassian.net/wiki/spaces/BORPub/pages/655369/Public+Available+Dataset you can download the tar ball for the small sample dataset.

## My Solution
----


### Semantic Segmentation on Kitti
----
- I have choosen Kitti Dataset for Vehicle Detection. you can find the description of dataset [here](http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015)
- I have tried to replicate [Fully Convolution Network](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
- I have used a pretrained VGG16 Model as Encoder and in Decoder I have used Transpose Convolution with Skip Connections from the encoder
- Download VGG16 from [here](http://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- Extract the contents of the VGG in the "./data/vgg_model" folder 
- This network is used for Semantic Segmentation whose architecture is below:
[image2]: ./image_rsrcs/semantic_segmentation.png "SemanticSegmentation"
![SemanticSegmentation][image2]

- Training image and its corresponding Ground Truth: 
[image3]: ./image_rsrcs/kitti_data.png "KittiData"
![KittiData][image3]


### Semantic Segmentation Results
----
[image4]: ./image_rsrcs/KittiInference.png "KittiInference"
![KittiInference][image4]

[image5]: ./image_rsrcs/KittiInference2.png "KittiInference2"
![KittiInference2][image5]


[image6]: ./image_rsrcs/KittiInference3.png "KittiInference3"
![KittiInference3][image6]


### Object Detection   
----

### Object Detection with Tensorflow API
----
- Used TensorFlow's Pretrained SSD model fro detecting Cars
- This model is capable of performing object detecting in real time which is faster than Semantic Segmentation
- Link to TensorFlow Object Detection Api is [here](https://github.com/tensorflow/models/tree/master/research/object_detection)
- This model is used to detect cars on this data: https://borgward.atlassian.net/wiki/spaces/BORPub/pages/655369/Public+Available+Datase


### Results
----
[image7]: ./image_rsrcs/carDetection1.png "carDetection1"
![carDetection1][image7]

[image8]: ./image_rsrcs/carDetection2.png "carDetection2"
![carDetection2][image8]


[image9]: ./image_rsrcs/carDetection3.png "carDetection3"
![carDetection3][image9]


