# Winnerineast的代码垃圾堆代码索引
这是用来解释我的整个github代码结构的网页，不会中文的朋友，请用谷歌翻译吧。

我的结构顺序如下，主要是根据我自己的研究和开发需要。
- [IoT应用类](#IoT应用类)
- [图像对象检测和实例分割算法类](#图像对象检测和实例分割算法类)
- [通用图像算法框架类](#通用图像算法框架类)
- [NLP自然语言处理算法类](#NLP自然语言处理算法类)
- [NLP自然语言处理框架类](#NLP自然语言处理框架类)
- [语音算法类](#语音算法类)
- [基于ARM处理器的深度学习框架类](#基于ARM处理器的深度学习框架类)
- [通用深度学习算法类](#通用深度学习算法类)
- [通用机器学习深度学习框架类](#通用机器学习深度学习框架类)
- [推荐系统类](#推荐系统类)
- [深度学习电子书类](#深度学习电子书类)
- 大数据工具和互联网技术
- [数据格式化类](#数据格式化类)
- [数据存储类](#数据存储类)
- [数据可视化类](#数据可视化类)
- [网络安全技术类](#网络安全技术类)

## <a name="IoT应用类">IoT应用类</a>
- [home-assistant](https://github.com/winnerineast/home-assistant)：一个集成监测和控制各种智能家居设备的python代码。适合于做各类监控系统的起步代码。

## <a name="图像对象检测和实例分割算法类">图像对象检测和实例分割算法类</a>
- [face_recognition](https://github.com/winnerineast/face_recognition)：这是一个基于[dlib](https://github.com/winnerineast/dlib)实现的人脸识别系统，精度据说达到99.38%，主打树莓派应用，是嵌入式人脸识别的很好入门产品和技术。
- [Mask_RCNN](https://github.com/winnerineast/Mask_RCNN)：这是大神何恺明的最新[论文](https://arxiv.org/abs/1703.06870)的python实验性实现代码。
- [mx-maskrcnn](https://github.com/winnerineast/mx-maskrcnn)：这是大神何恺明的最新[论文](https://arxiv.org/abs/1703.06870)的一种MXNET代码实现，基于另一种实现[mx-rcnn](https://github.com/winnerineast/mx-rcnn)。
- [FastMaskRCNN](https://github.com/winnerineast/FastMaskRCNN)：这是大神何恺明的最新[论文](https://arxiv.org/abs/1703.06870)的[tensorflow](https://github.com/winnerineast/tensorflow)实验性实现代码。
- [chainercv](https://github.com/winnerineast/chainercv)：基于[chainer](https://github.com/chainer/chainer)的算法实现。
- [luminoth](https://github.com/winnerineast/luminoth)：基于[tensorflow](https://github.com/winnerineast/tensorflow)和[sonnet](https://github.com/winnerineast/sonnet)的算法。
- [darknet](https://github.com/winnerineast/darknet)：卷积神经网络基于OpenCV的算法。

## <a name="通用图像算法框架类">通用图像算法框架类</a>
- [dlib](https://github.com/winnerineast/dlib)：大有超越OpenCV味道的，发展最快的计算机视觉库，特别是在支持C++和Python方面有自己的特色，值得自己做底层库的人去学习这种写法。
- [opencv](https://github.com/winnerineast/opencv)：最有名的计算机视觉处理库。

## <a name="NLP自然语言处理算法类">NLP自然语言处理算法类</a>
- [fastText](https://github.com/winnerineast/fastText)：脸书发布的文本表征和分类算法。
- [BayesianRecurrentNN](https://github.com/winnerineast/BayesianRecurrentNN)：这是论文[Bayesian Recurrent Neural Networks](https://arxiv.org/abs/1704.02798)的Tensorflow实现。

## <a name="NLP自然语言处理框架类">NLP自然语言处理框架类</a>
- [spaCy](https://github.com/winnerineast/spaCy)：一个强悍和快速的NLP python库，几乎各种传统的NLP功能都有，并且支持25种语言，其中8种语言自带13种统计模型，预训练的词向量，自带可视化语法解析树，底层是Cython，所以很快。
- [OpenNE](https://github.com/winnerineast/OpenNE):清华大学自然语言处理实验室发布的网络嵌入工具。
- [OpenKE](https://github.com/winnerineast/OpenKE)：清华大学自然语言处理实验室发布的知识嵌入工具。

## <a name="语音算法类">语音算法类</a>
- [mss_pytorch](https://github.com/winnerineast/mss_pytorch)：对于歌曲中人声进行增强，音乐弱化的算法。
- [multi-speaker-tacotron-tensorflow](https://github.com/winnerineast/multi-speaker-tacotron-tensorflow)：多人语音转写算法的[tensorflow](https://github.com/winnerineast/tensorflow)实现。
- [kaldi](https://github.com/winnerineast/kaldi)：语音转写组合算法集。
- [librosa](https://github.com/winnerineast/librosa)：比较全面的语音和音乐处理算法库。

## <a name="基于ARM处理器的深度学习框架类">基于ARM处理器的深度学习框架类</a>
- [uTensor](https://github.com/winnerineast/uTensor)：基于[tensorflow](https://github.com/winnerineast/tensorflow)的超级轻量级的深度学习推断框架，使用[Mbed CLI](https://github.com/ARMmbed/mbed-cli)来编译。

## <a name="通用深度学习算法类>通用深度学习算法类</a>
- [Python Algo Collections](https://github.com/winnerineast/Python)：很多基础算法的python
- [CapsNet-Tensorflow](https://github.com/winnerineast/CapsNet-Tensorflow)：这是Hinton大神最新胶囊理论的[tensorflow](https://github.com/winnerineast/tensorflow)实现代码。这个代码用MNIST数据集来说明其有效性。
- [CapsNet-Keras](https://github.com/winnerineast/CapsNet-Keras)：这是Hinton大神最新胶囊理论的[keras](https://github.com/winnerineast/keras)实现代码，这个代码用MNIST数据集，目前测试错误率小于0.4%。
- [pytorch-capsule](https://github.com/winnerineast/pytorch-capsule)：采用了Pytorch实现了Hinton大神最新胶囊理论的一部分。
- [xgboost](https://github.com/winnerineast/xgboost)：陈天奇的通用梯度上升算法，几乎大部分的分类都可以用。

## <a name="通用机器学习深度学习框架类">通用机器学习深度学习框架类</a>
- [Origae-6](https://github.com/winnerineast/Origae-6)：这是我基于[DIGITS](https://github.com/NVIDIA/DIGITS)改的通用算法平台。
- [DIGITS](https://github.com/NVIDIA/DIGITS)：英伟达自己出的深度学习平台，没有什么亮点，主要针对视频和图像，适合于工业快速开发的起点平台。
- [tensorflow](https://github.com/winnerineast/tensorflow)：全球第一的深度学习框架，它还带有一系列的[模型](https://github.com/winnerineast/models)，也有[Lattice](https://github.com/winnerineast/lattice)这样支持推荐系统的点阵模型。
- [keras](https://github.com/winnerineast/keras)：在Theano宣布死亡之后，拥抱Tensorflow的最佳快速编程框架，非常适合于验证各种论文算法。
- [caffe](https://github.com/winnerineast/caffe)：这个世界上最强大，最难用，最广泛使用的深度学习框架，现在投奔Facebook，主推[caffe2](https://github.com/winnerineast/caffe2)。
- [caffe2](https://github.com/winnerineast/caffe2)：由于caffe框架的难用，所以Facebook开始打造第二个版本。
- [scikit-learn](https://github.com/winnerineast/scikit-learn)：大名鼎鼎的机器学习算法库。
- [hw](https://github.com/winnerineast/hw)：英伟达的深度学习加速器用到的一些库，用GPU的需要了解。
- [pyro](https://github.com/winnerineast/pyro)：一种通用概率模型算法框架，来自优步人工智能实验室。
- [deepo](https://github.com/winnerineast/deepo)：一个集成了 theano, [tensorflow](https://github.com/winnerineast/tensorflow), sonnet, pytorch, [keras](https://github.com/winnerineast/keras), lasagne, mxnet, cntk, [chainer](https://github.com/chainer/chainer), caffe, torch的容器，主要用于各种云计算资源部署。
- [chainer](https://github.com/chainer/chainer)：一种通用深度学习框架，主打视觉领域。
- [pytorch-qrnn](https://github.com/winnerineast/pytorch-qrnn)：Quasi-RNN的pytorch代码实现，这是一种LSTM的替代品，但是速度是英伟达的cuDNN极致优化后LSTM的2-17倍。

## <a name="推荐系统类">推荐系统类</a>
- [librec](https://github.com/winnerineast/librec)：使用Java实现的完备的推荐系统，包含了从原始数据转化到训练数据集，相似度计算，训练，过滤，推荐的全过程。
- [predictionio](https://github.com/winnerineast/incubator-predictionio)：这是apache基于Spark和HBase的预测算法框架，适合于做推荐系统，相似度计算，分类算法。

## <a name="深度学习电子书类">深度学习电子书类</a>
- [tensorflow-mnist-tutorial](https://github.com/winnerineast/tensorflow-mnist-tutorial)：这是学习Tensorflow的人必须看的东西。谷歌大神Martin Gorner在2017年python大会上讲演用的代码，比较好的诠释了Tensorflow怎么应用于深度学习的，使用的是MNIST数据集。
- [deeplearningbook-chinese](https://github.com/winnerineast/deeplearningbook-chinese):这是一本深度学习英文版的中文翻译，这本书中规中矩，适合于希望做神经网络底层优化的人系统学习，也能给应用神经网络的人一些有益的知识。

## ***********************************************************************************
## 以下内容和深度学习关系不大，但是工程化和落地深度学习技术，这些技术和工具都还是很有用处的。
## ***********************************************************************************

## <a name="数据收集类">数据收集类</a>
- [scrapy](https://github.com/winnerineast/scrapy)：一个有名的python爬虫系统，可以结合一些Java Script库来爬动态页面。
- [ItChat](https://github.com/winnerineast/ItChat)：微信 python API，可以用来爬微信聊天记录，可以作为对话机器人的输出接口。

## <a name="数据格式化类">数据格式化类</a>
- [protobuf](https://github.com/winnerineast/protobuf)：谷歌的数据交换建模语言，方便python，java，c语言的数据交换。

## <a name="数据存储类">数据存储类</a>
- [leveldb](https://github.com/winnerineast/leveldb):谷歌用来存储图片，文档的Key-Value数据库，特别适合于存放打标签的数据。

## <a name="数据可视化类">数据可视化类</a>
- [manim](https://github.com/winnerineast/manim)：在Latex文档里面做公式动画，比较适合于解释比较困难的公式。
- [react](https://github.com/winnerineast/react)：毫无疑问，脸书希望打败手机原生应用的武器。
- [django](https://github.com/winnerineast/django)：可以用来做任何系统的快速开发原型，特别适合于人工智能系统。

## <a name="网络安全技术类">网络安全技术类</a>
- [AngelSword](https://github.com/winnerineast/AngelSword)：一个检测各类CMS漏洞的python代码，主要特色是支持很多CMS系统的接口渗透。
