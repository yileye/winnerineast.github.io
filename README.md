# Winnerineast的代码垃圾堆代码索引
这是用来解释我的整个github代码结构的网页，不会中文的朋友，请用谷歌翻译吧。
## IoT应用类
- [home-assistant](https://github.com/winnerineast/home-assistant)：一个集成监测和控制各种智能家居设备的python代码。适合于做各类监控系统的起步代码。

## 图像对象检测和实例分割算法类
- [Mask_RCNN](https://github.com/winnerineast/Mask_RCNN)：这是大神何恺明的最新[论文](https://arxiv.org/abs/1703.06870)的python实验性实现代码。
- [mx-maskrcnn](https://github.com/winnerineast/mx-maskrcnn)：这是大神何恺明的最新[论文](https://arxiv.org/abs/1703.06870)的一种MXNET代码实现，基于另一种实现[mx-rcnn](https://github.com/winnerineast/mx-rcnn)。
- [FastMaskRCNN](https://github.com/winnerineast/FastMaskRCNN)：这是大神何恺明的最新[论文](https://arxiv.org/abs/1703.06870)的[tensorflow](https://github.com/winnerineast/tensorflow)实验性实现代码。
- [chainercv](https://github.com/winnerineast/chainercv)：基于[chainer](https://github.com/chainer/chainer)的算法实现。
- [luminoth](https://github.com/winnerineast/luminoth)：基于[tensorflow](https://github.com/winnerineast/tensorflow)和[sonnet](https://github.com/winnerineast/sonnet)的算法。
- [darknet](https://github.com/winnerineast/darknet)：卷积神经网络基于OpenCV的算法。

## 文本算法类
- [fastText](https://github.com/winnerineast/fastText)：脸书发布的文本表征和分类算法。

## NLP自然语言处理算法类
- [BayesianRecurrentNN](https://github.com/winnerineast/BayesianRecurrentNN)：这是论文[Bayesian Recurrent Neural Networks](https://arxiv.org/abs/1704.02798)的Tensorflow实现。

## NLP自然语言处理框架类
- [spaCy](https://github.com/winnerineast/spaCy)：一个强悍和快速的NLP python库，几乎各种传统的NLP功能都有，并且支持25种语言，其中8种语言自带13种统计模型，预训练的词向量，自带可视化语法解析树，底层是Cython，所以很快。
- [OpenNE](https://github.com/winnerineast/OpenNE):清华大学自然语言处理实验室发布的网络嵌入工具。
- [OpenKE](https://github.com/winnerineast/OpenKE)：清华大学自然语言处理实验室发布的知识嵌入工具。

## 语音算法类
- [mss_pytorch](https://github.com/winnerineast/mss_pytorch)：对于歌曲中人声进行增强，音乐弱化的算法。
- [multi-speaker-tacotron-tensorflow](https://github.com/winnerineast/multi-speaker-tacotron-tensorflow)：多人语音转写算法的[tensorflow](https://github.com/winnerineast/tensorflow)实现。
- [kaldi](https://github.com/winnerineast/kaldi)：语音转写组合算法集。
- [librosa](https://github.com/winnerineast/librosa)：比较全面的语音和音乐处理算法库。

## 基于ARM处理器的深度学习框架类
- [uTensor](https://github.com/winnerineast/uTensor)：基于[tensorflow](https://github.com/winnerineast/tensorflow)的超级轻量级的深度学习推断框架，使用[Mbed CLI](https://github.com/ARMmbed/mbed-cli)来编译。

## 通用机器学习，深度学习框架类
- [Origae-6](https://github.com/winnerineast/Origae-6)：这是我基于[DIGITS](https://github.com/NVIDIA/DIGITS)改的通用算法平台。
- [tensorflow](https://github.com/winnerineast/tensorflow)：全球第一的深度学习框架，它还带有一系列的[模型](https://github.com/winnerineast/models)，也有[Lattice](https://github.com/winnerineast/lattice)这样支持推荐系统的点阵模型。
- [keras](https://github.com/winnerineast/keras)：在Theano宣布死亡之后，拥抱Tensorflow的最佳快速编程框架，非常适合于验证各种论文算法。
- [scikit-learn](https://github.com/winnerineast/scikit-learn)：大名鼎鼎的机器学习算法库。
- [pyro](https://github.com/winnerineast/pyro)：一种通用概率模型算法框架，来自优步人工智能实验室。
- [deepo](https://github.com/winnerineast/deepo)：一个集成了 theano, [tensorflow](https://github.com/winnerineast/tensorflow), sonnet, pytorch, [keras](https://github.com/winnerineast/keras), lasagne, mxnet, cntk, [chainer](https://github.com/chainer/chainer), caffe, torch的容器，主要用于各种云计算资源部署。
- [chainer](https://github.com/chainer/chainer)：一种通用深度学习框架，主打视觉领域。

## 通用深度学习算法类
- [CapsNet-Tensorflow](https://github.com/winnerineast/CapsNet-Tensorflow)：这是Hinton大神最新胶囊理论的[tensorflow](https://github.com/winnerineast/tensorflow)实现代码。这个代码用MNIST数据集来说明其有效性。
- [CapsNet-Keras](https://github.com/winnerineast/CapsNet-Keras)：这是Hinton大神最新胶囊理论的[keras](https://github.com/winnerineast/keras)实现代码，这个代码用MNIST数据集，目前测试错误率小于0.4%。
- [pytorch-capsule](https://github.com/winnerineast/pytorch-capsule)：采用了Pytorch实现了Hinton大神最新胶囊理论的一部分。
- [xgboost](https://github.com/winnerineast/xgboost)：陈天奇的通用梯度上升算法，几乎大部分的分类都可以用。

## 推荐系统类
- [librec](https://github.com/winnerineast/librec)：使用Java实现的完备的推荐系统，包含了从原始数据转化到训练数据集，相似度计算，训练，过滤，推荐的全过程。
- [predictionio](https://github.com/winnerineast/incubator-predictionio)：这是apache基于Spark和HBase的预测算法框架，适合于做推荐系统，相似度计算，分类算法。

## 深度学习电子书类
- [deeplearningbook-chinese](https://github.com/winnerineast/deeplearningbook-chinese):这是一本深度学习英文版的中文翻译，这本书中规中矩，适合于希望做神经网络底层优化的人系统学习，也能给应用神经网络的人一些有益的知识。

## **************************************************************************************************************************
## 以下内容和深度学习关系不大，但是工程化和落地深度学习技术，这些技术和工具都还是很有用处的。
## **************************************************************************************************************************

## 数据收集类
- [scrapy](https://github.com/winnerineast/scrapy)：一个有名的python爬虫系统，可以结合一些Java Script库来爬动态页面。
- [ItChat](https://github.com/winnerineast/ItChat)：微信 python API，可以用来爬微信聊天记录，可以作为对话机器人的输出接口。

## 数据格式化类
- [protobuf](https://github.com/winnerineast/protobuf)：谷歌的数据交换建模语言，方便python，java，c语言的数据交换。

## 数据可视化类
- [manim](https://github.com/winnerineast/manim)：在Latex文档里面做公式动画，比较适合于解释比较困难的公式。

## HTML5技术类
- [react](https://github.com/winnerineast/react)：毫无疑问，脸书希望打败手机原生应用的武器。

## 网络安全技术类
- [AngelSword](https://github.com/winnerineast/AngelSword)：一个检测各类CMS漏洞的python代码，主要特色是支持很多CMS系统的接口渗透。
