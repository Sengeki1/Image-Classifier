# Image-Classifier
A  simple image classification pipeline based on the k-nearest Neighbor or the SVM/Softmax classifier. 
The goals of this assignment are as follows:

- understand the basic Image Classification pipeline and the data-driven approach (train/predict stages)
- understand the train/val/test splits and the use of validation data for hyperparameter tuning.
- develop proficiency in writing efficient vectorized code with numpy
- implement and apply a k-Nearest Neighbor (kNN) classifier
- implement and apply a Multiclass Support Vector Machine (SVM) classifier
- implement and apply a Softmax classifier
- implement and apply a Two layer neural network classifier
- understand the differences and tradeoffs between these classifiers
- get a basic understanding of performance improvements from using higher-level representations than raw pixels (e.g. color histograms,
Histogram of Gradient (HOG) features).

We will use the CIFAR-10 dataset.

## How does KNN work?
Following is a spread of red circles (RC) and green squares (GS):

![image](https://github.com/Sengeki1/Image-Classifier/assets/106749775/c75da09e-91c7-4b5f-9153-59ff32ae50d5)

You intend to find out the class of the blue star (BS). BS can either be RC or GS and nothing else. The “K” in KNN algorithm is the nearest neighbor we wish to take the vote from. Let’s say K = 3. Hence, we will now make a circle with BS as the center just as big as to enclose only three data points on the plane.

![image](https://github.com/Sengeki1/Image-Classifier/assets/106749775/5fe2b10f-2bc0-4c3e-83a5-70fe0c7446b3)

The three closest points to BS are all RC. Hence, with a good confidence level, we can say that the BS should belong to the class RC. Here, the choice became obvious as all three votes from the closest neighbor went to RC. The choice of the parameter K is very crucial in this algorithm.

### How do we choose K?

![image](https://github.com/Sengeki1/Image-Classifier/assets/106749775/b794bae4-a26b-40cb-8065-95b9a6368e70)
![image](https://github.com/Sengeki1/Image-Classifier/assets/106749775/efb3d3fc-5ecb-4352-a91e-71e698379a80)


If you watch carefully, you can see that the boundary becomes smoother with increasing value of K. With K increasing to infinity it finally becomes all blue or all red depending on the total majority.  The training error rate and the validation error rate are two parameters we need to access different K-value. Following is the curve for the training error rate with a varying value of K :

![image](https://github.com/Sengeki1/Image-Classifier/assets/106749775/a46a9b77-89b4-472e-9312-e105e8f16360)

As you can see, the error rate at K=1 is always zero for the training sample. This is because the closest point to any training data point is itself.Hence the prediction is always accurate with K=1. If validation error curve would have been similar, our choice of K would have been 1. Following is the validation error curve with varying value of K:

![image](https://github.com/Sengeki1/Image-Classifier/assets/106749775/4575aa29-5ce2-4891-be1d-527e5099e3b9)
