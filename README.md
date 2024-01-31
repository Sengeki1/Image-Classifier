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


## How SVM Works?
Here, we have two points in two-dimensional space, we have two columns x1 and x2. And we have some observations such as red and green, which are already classified. This is linearly separable data.

![image](https://github.com/Sengeki1/Image-Classifier/assets/106749775/9413d273-15e6-4af1-929a-0ae2a998eef4)

But, now how do we derive a line that separates these points? This means a separation or decision boundary is very important for us when we add new points.
So to classify new points, we need to create a boundary between two categories, and when in the future we will add new points and we want to classify them, then we know where they belong. Either in a Green Area or Red Area.

### So how can we separate these points?

One way is to draw a vertical line between two areas, so anything on the right is Red and anything on the left is Green. Something like that-

![image](https://github.com/Sengeki1/Image-Classifier/assets/106749775/1aed4944-681c-4717-8bb6-ac94035f0ea4)

However, there is one more way, draw a horizontal line or diagonal line. You can create multiple diagonal lines, which achieve similar results to separate our points into two classes.
But our main task is to find the optimal line or best decision boundary. And for this SVM is used. SVM finds the best decision boundary, which helps us to separate points into different spaces.
SVM finds the best or optimal line through the maximum margin, which means it has max distance and equidistance from both classes or spaces. The sum of these two classes has to be maximized to make this line the maximum margin.

![image](https://github.com/Sengeki1/Image-Classifier/assets/106749775/4ba08176-0cd9-4edc-a526-70126769d3ea)

These, two vectors are support vectors. In SVM, only support vectors are contributing. That’s why these points or vectors are known as support vectors. Due to support vectors, this algorithm is called a Support Vector Algorithm(SVM).
In the picture, the line in the middle is a maximum margin hyperplane or classifier. In a two-dimensional plane, it looks like a line, but in a multi-dimensional, it is a hyperplane. That’s how SVM works.


## How Softmax work

Softmax regression (or multinomial logistic regression) is a generalization of logistic regression to the case where we want to handle multiple classes. In logistic regression we assumed that the labels were binary: y(i)∈{0,1} . We used such a classifier to distinguish between two kinds of hand-written digits.

![image](https://github.com/Sengeki1/Image-Classifier/assets/106749775/ed840a90-cde8-456f-b0f3-e08e9299c713)

As the name suggests, in softmax regression (SMR), we replace the sigmoid logistic function by the so-called softmax function φ:

![image](https://github.com/Sengeki1/Image-Classifier/assets/106749775/a0b2646d-96aa-41d9-9b14-2e73fd7960d4)

where we define the net input z as

![image](https://github.com/Sengeki1/Image-Classifier/assets/106749775/a83e3b83-0df1-4fa3-9b06-31e5721e5e7c)

(w is the weight vector, x is the feature vector of 1 training sample, and w0 is the bias unit.)
Now, this softmax function computes the probability that this training sample x(i) belongs to class j given the weight and net input z(i). So, we compute the probability p(y = j | x(i); wj) for each class label in j = 1, ..., k. Note the normalization term in the denominator which causes these class probabilities to sum up to one.


## How a Neural Network works

Neural networks are relatively crude electronic networks of neurons based on the neural structure of the brain. They process records one at a time, and learn by comparing their classification of the record (i.e., largely arbitrary) with the known actual classification of the record. The errors from the initial classification of the first record is fed back into the network, and used to modify the networks algorithm for further iterations.

![image](https://github.com/Sengeki1/Image-Classifier/assets/106749775/f8853231-702d-425b-9ce3-cec533c812c7)


Neurons are organized into layers: input, hidden and output. The input layer is composed not of full neurons, but rather consists simply of the record's values that are inputs to the next layer of neurons. The next layer is the hidden layer. Several hidden layers can exist in one neural network. The final layer is the output layer, where there is one node for each class. A single sweep forward through the network results in the assignment of a value to each output node, and the record is assigned to the class node with the highest value.

### Forward Propagation

![image](https://github.com/Sengeki1/Image-Classifier/assets/106749775/58b2f398-bae6-40d0-a7e1-a3231359b337)

Forward propagation is where input data is fed through a network, in a forward direction, to generate an output. The data is accepted by hidden layers and processed, as per the activation function, and moves to the successive layer. The forward flow of data is designed to avoid data moving in a circular motion, which does not generate an output. 

During forward propagation, pre-activation and activation take place at each hidden and output layer node of a neural network. The pre-activation function is the calculation of the weighted sum. The activation function is applied, based on the weighted sum, to make the neural network flow non-linearly using bias. 

### BackPropagation

Neural networks use supervised learning to generate output vectors from input vectors that the network operates on. It Compares generated output to the desired output and generates an error report if the result does not match the generated output vector. Then it adjusts the weights according to the bug report to get your desired output.

#### Backpropagation Algorithm:

- Step 1: Inputs X, arrive through the preconnected path.

- Step 2: The input is modeled using true weights W. Weights are usually chosen randomly.

- Step 3: Calculate the output of each neuron from the input layer to the hidden layer to the output layer.

- Step 4: Calculate the error in the outputs:

  ```Backpropagation Error= Actual Output – Desired Output```
  
- Step 5: From the output layer, go back to the hidden layer to adjust the weights to reduce the error.

- Step 6: Repeat the process until the desired output is achieved.

![image](https://github.com/Sengeki1/Image-Classifier/assets/106749775/dbac179b-a4e1-4296-b9fe-11734cd8e584)


For more explanation go to <https://towardsdatascience.com/understanding-backpropagation-algorithm-7bb3aa2f95fd> or <https://www.geeksforgeeks.org/backpropagation-in-data-mining/>


