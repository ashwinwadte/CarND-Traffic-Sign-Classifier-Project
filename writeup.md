# **Traffic Sign Recognition**

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/train_visualization.jpg "Visualization"
[image2]: ./examples/preprocessed_data.png "PreProcessed"
[image3]: ./examples/postprocessed_data.png "PostProcessed"
[image4]: ./traffic_sign_test_images/1.jpg "Traffic Sign 1"
[image5]: ./traffic_sign_test_images/2.jpg "Traffic Sign 2"
[image6]: ./traffic_sign_test_images/3.jpg "Traffic Sign 3"
[image7]: ./traffic_sign_test_images/4.jpg "Traffic Sign 4"
[image8]: ./traffic_sign_test_images/5.jpg "Traffic Sign 5"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the train data distribution with classes on X-axis and examples per class on Y-axis.

![Visualization][image1]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale as used in LeNet's Lab, it seems to give better accurancy than colored images.

Then, I normalized the image data because zero-mean and equal-variance gives better training performance.

Here is an example of a traffic sign image before and after processing.

![PreProcessed][image2]

![PostProcessed][image3]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 image   							    |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Flatten               | outputs 400                                   |
| Dropout               |                                               |
| Fully connected		| outputs 120        							|
| RELU                  |                                               |
| Fully connected		| outputs 84        							|
| RELU                  |                                               |
| Fully connected		| outputs 43        							|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the AdamOptimizer as applied in LeNet's Lab. Hyperparameters are as follows:

| Hyperparameter      		  |     Value	        					      |
|:---------------------------:|:---------------------------------------------:|
| Number of epochs            | 50   							              |
| Batch size     	          | 1024                          	              |
| Learning rate		    	  | 0.001									      |
| μ, σ	      	              | 0, 0.1 			                 	          |
| Keep probability	          | 0.5	                                          |


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.992
* validation set accuracy of 0.944
* test set accuracy of 0.929

As a first approach I played around with number of epochs, batch size and learning rate as in the LeNet's Lab. I was not getting validation accurancy above 0.91. Thus I added dropout and accuracy increased. Then I increased the batch size and learning rate. This way, I got the validation accuracy of over 0.93. The training set accuracy is 0.99 while the test set accurancy and validation set accurancy are nearly same, so the model does good job with real data.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Traffic Sign 1][image4] ![Traffic Sign 2][image5] ![Traffic Sign 3][image6]
![Traffic Sign 4][image7] ![Traffic Sign 5][image8]

The forth image might be difficult to classify because its little tilted.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			                        |     Prediction	        					|
|:-------------------------------------:|:---------------------------------------------:|
| Bumpy Road      		                | Bumpy Road   									|
| Right-of-way at the next intersection | Right-of-way at the next intersection         |
| Road work				                | Road work										|
| Children crossing	      		        | Right-of-way at the next intersection			|
| Speed limit (30km/h)			        | SSpeed limit (30km/h)      					|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 92.9%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Bumpy Road sign (probability of 0.99), and the image does contain a Bumpy Road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .99         			| Bumpy Road   									|
| .99     				| Right-of-way at the next intersection         |
| .99					| Road work										|
| .04	      			| Right-of-way at the next intersection			|
| .97				    | SSpeed limit (30km/h)      					|


For all the images, the probability is almost 1.0 except 4th image where probability was 0.04 for correct class.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


