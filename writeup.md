# **Traffic Sign Recognition** 

## This writeup file is created from Writeup Template provided by Udacity.
## All rights reserved by Udacity

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

[image1]: ./screenshots/original_images.png "Original"
[image2]: ./screenshots/augmented_images.png "Augmented"
[image3]: ./examples/placeholder.png "Traffic Sign 2"
[image4]: ./new_images_data/00004.jpg "Speed limit (70km/h)"
[image5]: ./new_images_data/00007.jpg "Speed limit (100km/h)"
[image6]: ./new_images_data/00011.jpg "Right-of-way at the next intersection"
[image7]: ./new_images_data/00013.jpg "Yield"
[image8]: ./new_images_data/00014.jpg "Stop"
[image9]: ./new_images_data/00017.jpg "No entry"
[image10]: ./new_images_data/00023.jpg "Slippery road"
[image11]: ./new_images_data/00025.jpg "Road work"
[image12]: ./new_images_data/00035.jpg "Ahead only"
[image13]: ./new_images_data/00040.jpg "Roundabout mandatory"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/ikcGitHub/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it can turn a 3 dimensional arrya to 1 dimensional array which can make the computation much easier.

As a last step, I normalized the image data because it can enhance the accuracy of the training model.

Here is an example of an augmented image:
There is an improvement here we can do in the future.
__[Improvement]: Improve the image dataset by applying the formula np.dot(X_train[:,:,:,:3], [0.299, 0.587, 0.114])__

![alt text][image2]




####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grey image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x6     									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Drop out       | keep_prob = 0.5     |
| Flatten      |     outputs 400
| Fully connected		|   outputs 120						|
| Fully connected		|   outputs 84						|
| Fully connected		|   outputs 43						|
|	Softmax					|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I followed the LeNet-5 to create my own training model based on it.
Here is a list the parameters

| Parameters | Value | description |
|:---------------------:|:---------------------------------------------:| :---------------------------------------------:| 
| EPOCHS | 40 | epochs |
| BATCH_SIZE | 128 | batch size |
| mu | 0 | average |
| sigma | 0.1 | standard deviation |
| rate | 0.001 | learning rate |




####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of 0.957
* test set accuracy of 0.947

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

My answer: here is my first architecture which comes from LeNet-5 since this would be a good starting point.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grey image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x6     									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten      |     outputs 400
| Fully connected		|   outputs 120						|
| Fully connected		|   outputs 84						|
| Fully connected		|   outputs 43						|
|	Softmax					|												|
|						|												|

* What were some problems with the initial architecture?

My answer: The accuracy is between 0.863 and 0.895. And I observed both the valid accuracy and test accuracy are low. Therefore, a underfitting might exist.


* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

My answer: Drop out regularation was added after the convolution layers before the flatten layer. With epoch = 40, it can provide high accuracy for both valid and test dataset.

* Which parameters were tuned? How were they adjusted and why?

My answer:  The following parameter were tuned

| Parameters | Initial value | Final value | Procedure description |
|:---------------:|:-----------------:| :---------------:|  :---------------:| 
| EPOCHS | 20 | 40 | Try differet epches like 60 and 80, and found 40 is good enough to have test accuracy above 0.93 |
| keep_prob | None | 0.5 | Add drop out regularization after convolution layers, and found 0.5 is good enough |
| rate | 0.001 | 0.001 | Try different learning rates like 0.0005 and 0.0001, and found 0.001 is good enough once we have the drop out regularization |

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

My anwser: convolution layer can used to capture the charactrer of the images without losing too much information. And a dropout layer can help the model not underfitting/overfitting on the training dataset sicne it can randomly select different dataset to contribute the same result.

If a well known architecture was chosen:
* What architecture was chosen?

My anwser: LeNet-5

* Why did you believe it would be relevant to the traffic sign application?

My anwser: Because it is the first one I knew and learned.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
 My anwser: Yes, they are working well since the accuracy of validation and test dataset are higher than 0.93 and the number are almost equal to each other.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10] ![alt text][image11] ![alt text][image12]
![alt text][image13]

The 9th image might be difficult to classify because there is shadow on the sign.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)      		| Speed limit (70km/h)   									| 
| Speed limit (100km/h)     			| Speed limit (100km/h) 										|
| Right-of-way at the next intersection					| Right-of-way at the next intersection											|
| Yield	      		| Yield					 				|
| Stop			| Stop      							|
| No entry | No entry |
| Slippery road | Slippery road |
| Road work | General caution |
| Ahead only | Ahead only |
| Roundabout mandatory | Go straight or left |

The model was able to correctly guess 8 of the 10 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 93.8%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 19th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a "Speed limit (70km/h)" sign (probability of 0.999986), and the image does contain a "Speed limit (70km/h)" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.999986         			| Speed limit (70km/h)   									| 
| 1.0352e-05     				| Stop 										|
| 3.15043e-06					| Speed limit (30km/h)											|
| 9.63804e-08	      			| Speed limit (120km/h)					 				|
| 5.52293e-08				    | Speed limit (100km/h)      							|


For the second image, the model is relatively sure that this is a "Speed limit (100km/h)" sign (probability of 0.909949), and the image does contain a "Speed limit (100km/h)" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.909949         			| Speed limit (100km/h)   									| 
| 0.0336627     				| No passing for vehicles over 3.5 metric tons 										|
| 0.0273038					| Speed limit (120km/h)											|
| 0.0224156	      			| No passing					 				|
| 0.00596595				    | Vehicles over 3.5 metric tons prohibited      							|

For the third image, the model is relatively sure that this is a "Right-of-way at the next intersection" sign (probability of 0.996731), and the image does contain a "Right-of-way at the next intersection" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.996731         			| Right-of-way at the next intersection   									| 
| 0.0023167     				| Beware of ice/snow 										|
| 0.00067211					| Double curve											|
| 0.000115129	      			| Pedestrians					 				|
| 5.14639e-05				    | Slippery road      							|

For the fourth image, the model is relatively sure that this is a "Yield" sign (probability of 1.0), and the image does contain a "Yield" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Yield   									| 
| 1.01012e-23     				| Ahead only 										|
| 2.20637e-26					| Road work											|
| 3.48863e-28	      			| Priority road					 				|
| 7.84227e-29				    | Speed limit (30km/h)      							|

For the fifth image, the model is relatively sure that this is a "Stop" sign (probability of 0.99996), and the image does contain a "Stop" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99996         			| Stop   									| 
| 1.26435e-05     				| Keep right 										|
| 5.91276e-06					| No entry											|
| 4.96162e-06	      			| No vehicles					 				|
| 4.14061e-06				    | Speed limit (60km/h)      							|

For the sixth image, the model is relatively sure that this is a "No entry" sign (probability of 1.0), and the image does contain a "No entry" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| No entry   									| 
| 9.05509e-08     				| Stop 										|
| 2.20622e-10					| No passing											|
| 2.51324e-12	      			| No passing for vehicles over 3.5 metric tons					 				|
| 9.71079e-13				    | Keep right      							|

For the seventh image, the model is relatively sure that this is a "Slippery road" sign (probability of 0.567795), and the image does contain a "Slippery road" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.567795         			| Slippery road   									| 
| 0.42512     				| Dangerous curve to the left 										|
| 0.00704455					| Dangerous curve to the right											|
| 3.61435e-05	      			| Beware of ice/snow					 				|
| 4.66116e-06				    | Right-of-way at the next intersection      							|

For the eighth image, the model is relatively sure that this is a "General caution" sign (probability of 1.0), however, the image contains a "Road work" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| General caution   									| 
| 4.01071e-11     				| Traffic signals 										|
| 1.04885e-11					| Pedestrians											|
| 3.73492e-15	      			| Right-of-way at the next intersection					 				|
| 1.22789e-20				    | Children crossing      							|

For the ninth image, the model is relatively sure that this is a "Ahead only" sign (probability of 1.0), and the image does contain a "Ahead only" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Ahead only   									| 
| 2.57161e-08     				| No vehicles 										|
| 1.33408e-08					| Turn right ahead											|
| 8.86282e-09	      			| No passing					 				|
| 6.03572e-09				    | Turn left ahead      							|

For the tenth image, the model is relatively sure that this is a "General caution" sign (probability of 0.997141), however, the image contains a "Roundabout mandatoryk" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Go straight or left   									| 
| 0.00285933     				| Roundabout mandatory 										|
| 1.51674e-08					| Keep left											|
| 4.03953e-09	      			| Speed limit (20km/h)					 				|
| 1.12457e-10				    | Speed limit (30km/h)      							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


