# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/visualization.jpg "Visualization"
[image2]: ./writeup_images/bar_train.jpg "bar_train"
[image3]: ./writeup_images/bar_valid.jpg "bar_valid"
[image4]: ./writeup_images/bar_test.jpg "bar_test"
[image5]: ./writeup_images/loss.jpg "loss"
[image6]: ./writeup_images/new_pictures.jpg "5 pictures"
[image7]: ./writeup_images/loss2.jpg "loss2"
[image8]: ./writeup_images/bar_train_modify.jpg "bar_test_modify"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it! and here is a link to my [project code](./Traffic_Sign_Classifier.ipynb) .

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. 

I used the Python library to calculate summary statistics of the traffic
signs data set:

* The size of training set is `34799`
* The size of the validation set is `4410`
* The size of test set is `12630`
* The shape of a traffic sign image is `(32, 32, 3)`
* The number of unique classes/labels in the data set is `43`

#### 2. Include an exploratory visualization of the dataset.
First, randomly plot some traffic sign images.
![Visualization of the dataset][image1]

Then, plot the count of each sign.Here are three exploratory visualization of the data set. 
Thses are  bar charts showing how the data distribute among 43 classes in train, valid and test set.

![alt text][image2]
![alt text][image3]
![alt text][image4]

### Design and Test a Model Architecture

#### 1. Preprocessed the image data. 
As is shown in the visualization of the train dataset, the data are very imbalanced. So We could pick out the class label whose number of images less than 1000. Then apply rotation 90,180 and 270 and histogram equalization techniques on the image and add this pre-processed image to the dataset as “additional training set”. Here is the distribution of augmentation train data set.
![alt text][image8]

Let's begin to process data set.As a first step, I decided to convert every image to grayscale image. So the shape of input data is `[batch_size x 32*32]`.

Then  use `Min-Max Normalization` algorithm to process data. 

When return data, we should reshape data into `[batch_size x 32 x 32]`.

Here is the code.

```
def min_max_process(images):
    images_flatten=images.reshape(images.shape[0],-1)
    maxn=np.max(images_flatten,axis=1,keepdims=True)
    minn=np.min(images_flatten,axis=1,keepdims=True)
    result=(images_flatten-minn)/(maxn-minn)
    return result.reshape(result.shape[0],32,32,1)
```

#### 2. Model Architecture
My final model based on the LeNet-5, which consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   								| 
| Convolution 5x5x6     	| 1x1 stride, valid padding, outputs 28x28x6, activation= 'RELU' 	|
| Max pooling 2x2x6	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5x16	    | 1x1 stride, valid padding, outputs 10x10x16, activation= 'RELU'     		|
| Max pooling 2x2x16	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten	| outputs 400 |
| Dropout 	| keep_prob=0.4	|
| Fully connected		| outputs 120, activation= 'RELU'        									|
| Dropout 	| keep_prob=0.4	|
| Fully connected		| outputs 84, activation= 'RELU'        									|
| Fully connected		| outputs 43       									|


#### 3. Trained the model. 

To train the model, I used AdamOptimizer algotithm.

The hyperparameters I chose are following:

|	Hyperparameters 	| Value 	|
|:------------:|:-----:|
|	Learning rate 	| 0.003 	|
| 	Batch size 	|	128 |
| 	Epochs 	| 50	|

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy is `0.995` .
* validation set accuracy is `0.964` . 
* test set accuracy is `0.936` .
When training the model, save train-loss and valid-loss in each epoch.At first, I choose original LeNet-5 architecture. Although a high accuracy on the training set, it has low accuracy on the validation set. 
![alt text][image5]

Obviously, the initial architecture has overfitting problem. So adding dropout layers to avoid overfitting.I add two dropout layers at the first two fully-connect layers. Set the `keep_prob=0.4` .

![alt text][image7]

As we can see, the model in training set converge after 50 epoches(batch size =128). The validation set accuracy is great than 0.93. And the loss of validation set also become smaller, it mean that the model suitable for the current problem.
```
Training...

EPOCH 1 ...
Train Accuracy = 0.833
Validation Accuracy = 0.799

EPOCH 6 ...
Train Accuracy = 0.976
Validation Accuracy = 0.939
...
...
EPOCH 46 ...
Train Accuracy = 0.994
Validation Accuracy = 0.958

EPOCH 50 ...
Train Accuracy = 0.995
Validation Accuracy = 0.964

Model saved
```

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] 

The first image might be difficult to classify because the difference between speed limit traffic signs only is the number in the center.For the second picture, background color is similar to the traffic sign color. It may cause difficult to classify.The third image is a Stop sign, which shape is quiet unique from others. And the fourth image is very similar with the speed limit signs.As for the last picture, the Brightness of the image is low. So I think it is hard to tell the pattern in the center.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)      		| Speed limit (70km/h)  									| 
| Turn right ahead     			| Turn right ahead										|
| Stop 					| Stop 									|
| No vehicles	      		| No vehicles					 				|
| Traffic signals			| General caution     							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 

This compares favorably to the accuracy on the test set of `0.936` . The only mis-classification is **Traffic signals**. Let's disscus more deatil later.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


For the first image, the model is relatively sure that this is a Speed limit (70km/h)(probability of 0.99), and the image does contain a Speed limit (70km/h) sign. The top five soft max probabilities were all speed limit sign.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         			| Speed limit (70km/h)   									| 
| 0.0     				| Speed limit (20km/h) 										|
| 0.0					| Speed limit (30km/h)										|
| 0.0	      			| Speed limit (50km/h)					 				|
| 0.0				    | Speed limit (80km/h)    							|


For the second image, the prediction is true. Diffenence between top three class is relativelly small. It mean that the model is quiet confuse to tell us which traffic it is. as long as the model been modified, the accurancy would improve.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.22         			| Turn right ahead   									| 
| 0.21    				| Turn left ahead 										|
| 0.20					| Ahead only										|
| 0.11	      			| Go straight or right					 				|
| 0.09				    | Stop     							|

For the third image, the answer is certain. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         			| Stop   									| 
| 0.0    				| Speed limit (20km/h) 										|
| 0.0					| Go straight or left										|
| 0.0    			| Children crossing					 				|
| 0.0				    | Yield   	|

For the fourth image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.83         			| No vehicles   									| 
| 0.12   				| Roundabout mandatory 										|
| 0.02					| End of no passing										|
| 0.02	      			| Speed limit (50km/h)					 				|
| 0.01				    | Priority road |

The conclusion for the fifth image is wrong. The true answer is Traffic signals, which is the second high probablity. As long as the model been modified, the accurancy would improve.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.42         			| General caution  									| 
| 0.40    				| Traffic signals 										|
| 0.16					| Road narrows on the right										|
| 0.0	      			| Bicycles crossing				 				|
| 0.0				    | Pedestrians |

For the fifth image, I find an interesting thing. Carefully observe top five results, they have one common point. Yes! The shape of five signs are all triangle, and difference are patterns in the center. It may explain why this Convnet does misunderstand. Each picture has 32x32 pixel,so the quality of the images is low. That's why the model make a wrong prediction.

However, it may give us another perspective to understand CNN model(intuition). First detect the low level things such as the shape, and then detect the high level thing such as number, patterns and etc.


