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

Then, plott the count of each sign.Here are three exploratory visualization of the data set. 
Thses are  bar charts showing how the data distribute among 43 classes in train, valid and test set.

![alt text][image2]
![alt text][image3]
![alt text][image4]

### Design and Test a Model Architecture

#### 1. Preprocessed the image data. 

As a first step, I decided to convert every image to a row vector. So the shape of input data is `[batch_size x 32*32*3]`.

Then  use `Min-Max Normalization` algorithm to process data. 

When return data, we should reshape data into `[batch_size x 32 x 32 x3]`.

Here is the code.

```
def min_max_process(images):
    images_flatten=images.reshape(images.shape[0],-1)
    maxn=np.max(images_flatten,axis=1,keepdims=True)
    minn=np.min(images_flatten,axis=1,keepdims=True)
    result=(images_flatten-minn)/(maxn-minn)
    return result.reshape(result.shape[0],32,32,3)
```

#### 2. Model Architecture
My final model based on the LeNet-5, which consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------------:| 
| Input         		| 32x32x3 RGB image   								| 
| Convolution 5x5x6     	| 1x1 stride, valid padding, outputs 28x28x6, activation= 'RELU' 	|
| Max pooling 2x2x6	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5x16	    | 1x1 stride, valid padding, outputs 10x10x16, activation= 'RELU'     		|
| Max pooling 2x2x16	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten	| outputs 400 |
| Dropout 	| keep_prob=0.5	|
| Fully connected		| outputs 120, activation= 'RELU'        									|
| Dropout 	| keep_prob=0.5	|
| Fully connected		| outputs 84, activation= 'RELU'        									|
| Fully connected		| outputs 43       									|
 


#### 3. Trained the model. 

To train the model, I used AdamOptimizer algotithm.

The hyperparameters I chose are following:

|	Hyperparameters 	| Value 	|
|:------------:|:-----:|
|	Learning rate 	| 0.003 	|
| 	Batch size 	|	512 |
| 	Epochs 	| first:100 	final:40	|

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy is `0.997` .
* validation set accuracy is `0.934` . 
* test set accuracy is `0.909` .

At first, I choose original LeNet-5 architecture. Although a high accuracy on the training set, it has low accuracy on the validation set. Obviously, the initial architecture has overfitting problem. So adding dropout layers to avoid overfitting.

I add two dropout layers at the first two fully-connect layers. Set the `keep_prob=0.5` .

When training the model, save train-loss and valid-loss in each epoch. 
![alt text][image5]

As we can see, the model in training set converge after 40 epoches(batch size =100). The validation set accuracy is great than 0.93. But the loss of validation set became bigger after 40, it mean that the model has high variance. To avoid this problem, set `batch size =40`(early stop) and retrain the model.
```
Training...

EPOCH 1 ...
Train Accuracy = 0.796
Validation Accuracy = 0.723

EPOCH 6 ...
Train Accuracy = 0.987
Validation Accuracy = 0.907

EPOCH 11 ...
Train Accuracy = 0.995
Validation Accuracy = 0.910

EPOCH 16 ...
Train Accuracy = 0.997
Validation Accuracy = 0.919

EPOCH 21 ...
Train Accuracy = 0.996
Validation Accuracy = 0.910

EPOCH 26 ...
Train Accuracy = 0.999
Validation Accuracy = 0.944

EPOCH 31 ...
Train Accuracy = 0.999
Validation Accuracy = 0.944

EPOCH 36 ...
Train Accuracy = 0.995
Validation Accuracy = 0.920

EPOCH 40 ...
Train Accuracy = 0.997
Validation Accuracy = 0.934

Model saved
```
![alt text][image7]

So the architecture is suitable for the current problem. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] 

The first image might be difficult to classify because the difference between speed limit traffic signs only is the number in the center. And the fourth image is very similar with the speed limit signs.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)      		| Speed limit (70km/h)  									| 
| Turn right ahead     			| Ahead only										|
| Stop 					| Stop 									|
| No vehicles	      		| No vehicles					 				|
| Traffic signals			| Traffic signals     							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 

This compares favorably to the accuracy on the test set of `0.909` . The only mis-classification is **Turn right ahead**. Let's disscus more deatil later.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Speed limit (70km/h)(probability of 0.99), and the image does contain a Speed limit (70km/h) sign. The top five soft max probabilities were all speed limit sign.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99998569e-01         			| Speed limit (70km/h)   									| 
| 1.43504178e-06     				| Speed limit (30km/h) 										|
| 2.15403695e-08					| Speed limit (30km/h)										|
| 0.00000000e+00	      			| Speed limit (50km/h)					 				|
| 0.00000000e+00				    | Speed limit (60km/h)    							|


For the second image, the prediction is wrong. The right answer is  Turn right ahead sign, which is the second high probability. Diffenence between top two class is relativelly small. It mean that as long as the model been modified, the accurancy would improve.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.86804366e-01         			| Ahead only   									| 
| 1.29627772e-02    				| Turn right ahead 										|
| 2.32744147e-04					| Turn left ahead										|
| 1.15735980e-13	      			| Roundabout mandatory					 				|
| 4.60432156e-31				    | Vehicles over 3.5 metric tons prohibited     							|

For the third image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.83094871e-01         			| Stop   									| 
| 1.68127958e-02    				| Yield 										|
| 6.34110474e-05					| No passing for vehicles over 3.5 metric tons										|
| 1.87990863e-05	      			| Speed limit (60km/h)					 				|
| 1.01455062e-05				    | Road work   	|

For the fourth image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00         			| No vehicles   									| 
| 1.01447008e-22    				| Speed limit (70km/h) 										|
| 9.58415279e-26					| Yield										|
| 9.54842170e-27	      			| Speed limit (30km/h)					 				|
| 4.93103162e-27				    | No passing  |

For the fifth image, I find an interesting thing. The probability of the Traffic signals is about 100% , which has a high confidence. But carefully observe top five results, they have one common point. Yes! The shape of five signs are all triangle, and difference are patterns in the center. 

It may give us another perspective to understand CNN model(intuition). First detect the low level things such as the shape, and then detect the high level thing such as number, patterns and etc.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00         			| Traffic signals  									| 
| 3.00397573e-09    				| Road narrows on the right 		
								|
| 4.93576069e-10					| Bumpy road										|
| 5.23642286e-17	      			| Slippery road				 				|
| 6.24494182e-30				    | General caution  |



