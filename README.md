# BP-Iris
BP neural network is used to train and recognize three kinds of iris images, and the recognition rate is as high as 95%. However, all images will be grayscale before training.
If you know the learning rate adaptive algorithm, the speed of training will be greatly accelerated.
The current activation function is easy to cause the gradient to disappear and the calculation is complex, so we can choose the appropriate activation function instead.

这是用BP神经网络模型对三种不虹膜图像进行识别的代码，识别率高达95%。遗憾的是，因为之前训练的图像有灰度图，所以我统一的把所有图都转换成灰度图，虽然减轻了训练数据，但是同样的也减少了可区分的凭证。
现在的激活函数计算复杂且容易存在梯度消失的问题，你可以选择合适的激活函数进行替换。
学习率的调整十分麻烦，但你如果会学习率自适应算法，将大大的加快训练速度。
