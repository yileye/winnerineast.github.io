# Winnerineast的代码垃圾堆代码索引
这是用来解释我的整个github代码结构的网页，不会中文的朋友，请用谷歌翻译吧。

我的结构顺序如下，主要是根据我自己的研究和开发需要。
- [IoT应用类](#IoT应用类)
- [图像对象检测和实例分割算法类](#图像对象检测和实例分割算法类)
- [通用图像算法类](#通用图像算法类)
- [通用图像算法框架类](#通用图像算法框架类)
- [NLP自然语言处理算法类](#NLP自然语言处理算法类)
- [NLP自然语言处理框架类](#NLP自然语言处理框架类)
- [语音算法类](#语音算法类)
- [基于ARM处理器的深度学习框架类](#基于ARM处理器的深度学习框架类)
- [通用机器学习深度学习算法类](#通用机器学习深度学习算法类)
- [通用机器学习深度学习框架类](#通用机器学习深度学习框架类)
- [推荐系统类](#推荐系统类)
- [深度学习电子书类](#深度学习电子书类)
- 大数据工具和互联网技术
- [数据集](#数据集)
- [数据格式化类](#数据格式化类)
- [数据存储类](#数据存储类)
- [数据可视化类](#数据可视化类)
- [网络安全技术类](#网络安全技术类)
- [其它](#其它)

## <a name="IoT应用类">IoT应用类</a>
- [home-assistant](https://github.com/winnerineast/home-assistant)：一个集成监测和控制各种智能家居设备的python代码。适合于做各类监控系统的起步代码。

## <a name="通用图像算法类">通用图像算法类</a>
- [pix2pix](https://github.com/winnerineast/pix2pix)：图片风格迁移算法。
- [DenseNet](https://github.com/winnerineast/DenseNet)：2017年CVPR最佳论文的代码，主打图片语义理解，或者分类，对象识别等等。

## <a name="图像对象检测和实例分割算法类">图像对象检测和实例分割算法类</a>
- [face_recognition](https://github.com/winnerineast/face_recognition)：这是一个基于[dlib](https://github.com/winnerineast/dlib)实现的人脸识别系统，精度据说达到99.38%，主打树莓派应用，是嵌入式人脸识别的很好入门产品和技术。
- [Mask_RCNN](https://github.com/winnerineast/Mask_RCNN)：这是大神何恺明的最新[论文](https://arxiv.org/abs/1703.06870)的python实验性实现代码。
- [mx-maskrcnn](https://github.com/winnerineast/mx-maskrcnn)：这是大神何恺明的最新[论文](https://arxiv.org/abs/1703.06870)的一种MXNET代码实现，基于另一种实现[mx-rcnn](https://github.com/winnerineast/mx-rcnn)。
- [FastMaskRCNN](https://github.com/winnerineast/FastMaskRCNN)：这是大神何恺明的最新[论文](https://arxiv.org/abs/1703.06870)的[tensorflow](https://github.com/winnerineast/tensorflow)实验性实现代码。
- [chainercv](https://github.com/winnerineast/chainercv)：基于[chainer](https://github.com/chainer/chainer)的算法实现。
- [luminoth](https://github.com/winnerineast/luminoth)：基于[tensorflow](https://github.com/winnerineast/tensorflow)和[sonnet](https://github.com/winnerineast/sonnet)的算法。
- [darknet](https://github.com/winnerineast/darknet)：卷积神经网络基于OpenCV的算法。一大好消息，[darkflow](https://github.com/winnerineast/darkflow)能够把darknet的模型转换成tensorflow的模型。
- [ExtendedTinyFaces](https://github.com/winnerineast/ExtendedTinyFaces)：一种数照片里面人头的有趣算法。
- [FCIS](https://github.com/winnerineast/FCIS)：全卷积语义分割算法。
- [keras-retinanet](https://github.com/winnerineast/keras-retinanet)：keras构建的RetinaNet。
- [caffe-fast-rcnn](https://github.com/winnerineast/caffe-fast-rcnn)：Caffe构建的快速RCNN。

## <a name="通用图像算法框架类">通用图像算法框架类</a>
- [dlib](https://github.com/winnerineast/dlib)：大有超越OpenCV味道的，发展最快的计算机视觉库，特别是在支持C++和Python方面有自己的特色，值得自己做底层库的人去学习这种写法。
- [opencv](https://github.com/winnerineast/opencv)：最有名的计算机视觉处理库。
- [openpose](https://github.com/winnerineast/openpose)：CMU的有名的人体姿态追踪。

## <a name="NLP自然语言处理算法类">NLP自然语言处理算法类</a>
- [fastText](https://github.com/winnerineast/fastText)：脸书发布的文本表征和分类算法。
- [BayesianRecurrentNN](https://github.com/winnerineast/BayesianRecurrentNN)：这是论文[Bayesian Recurrent Neural Networks](https://arxiv.org/abs/1704.02798)的Tensorflow实现。
- [Synonyms](https://github.com/winnerineast/Synonyms)：中文近义词工具包。
- [cnn-text-classification-tf](https://github.com/winnerineast/cnn-text-classification-tf)：使用最成熟的文本分类算法。
- [keras-text](https://github.com/winnerineast/keras-text)：使用Keras构建的文本分类算法，包含了CNN的文本分类算法，RNN分类算法和Attention的RNN。
- [text_classification](https://github.com/winnerineast/text_classification)：文本分类汇总。
- [text-classification-cnn-rnn](https://github.com/winnerineast/text-classification-cnn-rnn)：CNN和RNN的中文文本分类。
- [MUSE](https://github.com/winnerineast/MUSE)：脸书的多语言嵌入训练。
- [Information-Extraction-Chinese](https://github.com/winnerineast/Information-Extraction-Chinese)：中文实体识别与关系提取。
- [flashtext](https://github.com/winnerineast/flashtext)：从句子中提取词语或者替换词语，特别是针对大规模文本提取和替换的性能极佳。
- [MatchZoo](https://github.com/winnerineast/MatchZoo)：文本匹配算法合集（DRMM, MatchPyramid, MV-LSTM, aNMM, DUET, ARC-I, ARC-II, DSSM, and CDSSM）。
- [kcws](https://github.com/winnerineast/kcws)：中文分词的深度学习算法。
- [Char-RNN-TensorFlow-Chinese](https://github.com/winnerineast/Char-RNN-TensorFlow-Chinese)：一个基于最新版本TensorFlow的Char RNN实现。可以实现生成英文、写诗、歌词、小说、生成代码、生成日文等功能。
- [paragraph-vectors](https://github.com/winnerineast/paragraph-vectors)：doc2vec的脸书实现。
- [fairseq-py](https://github.com/winnerineast/fairseq-py)：脸书的seq2seq实现。
- [LM-LSTM-CRF](https://github.com/winnerineast/LM-LSTM-CRF)：英文文本单词实体训练模型。

## <a name="NLP自然语言处理框架类">NLP自然语言处理框架类</a>
- [spaCy](https://github.com/winnerineast/spaCy)：一个强悍和快速的NLP python库，几乎各种传统的NLP功能都有，并且支持25种语言，其中8种语言自带13种统计模型，预训练的词向量，自带可视化语法解析树，底层是Cython，所以很快。
- [OpenNE](https://github.com/winnerineast/OpenNE):清华大学自然语言处理实验室发布的网络嵌入工具。
- [OpenKE](https://github.com/winnerineast/OpenKE)：清华大学自然语言处理实验室发布的知识嵌入工具。
- [FoolNLTK](https://github.com/winnerineast/FoolNLTK):一个中文处理包。
- [fnlp](https://github.com/winnerineast/fnlp)：中文自然语言处理工具包。
- [sling](https://github.com/winnerineast/sling)：谷歌的文本语义训练解析器。
- [allennlp](https://github.com/winnerineast/allennlp)：英文自然语言处理包。
- [cnschema](https://github.com/winnerineast/cnschema)：中文知识图谱。

## <a name="语音算法类">语音算法类</a>
- [mss_pytorch](https://github.com/winnerineast/mss_pytorch)：对于歌曲中人声进行增强，音乐弱化的算法。
- [multi-speaker-tacotron-tensorflow](https://github.com/winnerineast/multi-speaker-tacotron-tensorflow)：多人语音转写算法的[tensorflow](https://github.com/winnerineast/tensorflow)实现。
- [kaldi](https://github.com/winnerineast/kaldi)：语音转写组合算法集。
- [librosa](https://github.com/winnerineast/librosa)：比较全面的语音和音乐处理算法库。
- [wav2letter](https://github.com/winnerineast/wav2letter)：脸书开源ASRT。
- [DeepSpeech](https://github.com/winnerineast/DeepSpeech)：百度的ASR。
- [mss_pytorch](https://github.com/winnerineast/mss_pytorch)：歌曲人声分离算法，带demo页面。
- [tensorflow_end2end_speech_recognition](https://github.com/winnerineast/tensorflow_end2end_speech_recognition)：一种Tensorflow的CTC+Attention的ASR。

## <a name="基于ARM处理器的深度学习框架类">基于ARM处理器的深度学习框架类</a>
- [uTensor](https://github.com/winnerineast/uTensor)：基于[tensorflow](https://github.com/winnerineast/tensorflow)的超级轻量级的深度学习推断框架，使用[Mbed CLI](https://github.com/ARMmbed/mbed-cli)来编译。

## <a name="通用机器学习深度学习算法类">通用机器学习深度学习算法类</a>
- [Python Algo Collections](https://github.com/winnerineast/Python)：很多基础算法的python
- [CapsNet-Tensorflow](https://github.com/winnerineast/CapsNet-Tensorflow)：这是Hinton大神最新胶囊理论的[tensorflow](https://github.com/winnerineast/tensorflow)实现代码。这个代码用MNIST数据集来说明其有效性。
- [CapsNet-Keras](https://github.com/winnerineast/CapsNet-Keras)：这是Hinton大神最新胶囊理论的[keras](https://github.com/winnerineast/keras)实现代码，这个代码用MNIST数据集，目前测试错误率小于0.4%。
- [pytorch-capsule](https://github.com/winnerineast/pytorch-capsule)：采用了Pytorch实现了Hinton大神最新胶囊理论的一部分。
- [xgboost](https://github.com/winnerineast/xgboost)：陈天奇的通用梯度上升算法，几乎大部分的分类都可以用。
- [Paddle model](https://github.com/winnerineast/models-paddle):百度Paddle自带的模型，推荐类的比较深刻。
- [TVM](https://github.com/winnerineast/tvm):Tensor Intermediate Representation stack，自动生成硬件优化内核，来自陈天奇团队。
- [blocksparse](https://github.com/winnerineast/blocksparse)：专门针对英伟达GPU的性能优化。
- [Keras-Classification-Models](https://github.com/winnerineast/Keras-Classification-Models):各种用keras实现的分类算法合集。
- [sru](https://github.com/winnerineast/sru)：训练RNN和CNN一样快的优化，主要用于文本分类和语音识别。
- [SMASH](https://github.com/winnerineast/SMASH)：探索深度学习框架的工具。

## <a name="通用机器学习深度学习框架类">通用机器学习深度学习框架类</a>
- [Origae-6](https://github.com/winnerineast/Origae-6)：这是我基于[DIGITS](https://github.com/NVIDIA/DIGITS)改的通用算法平台。
- [DIGITS](https://github.com/NVIDIA/DIGITS)：英伟达自己出的深度学习平台，没有什么亮点，主要针对视频和图像，适合于工业快速开发的起点平台。
- [jetson-inference](https://github.com/winnerineast/jetson-inference)：基于Jetson TX1/TX2的深度学习平台，基于DIGITS。
- [tensorflow](https://github.com/winnerineast/tensorflow)：全球第一的深度学习框架，它还带有一系列的[模型](https://github.com/winnerineast/models)，也有[Lattice](https://github.com/winnerineast/lattice)这样支持推荐系统的点阵模型。还有[标准工程模版](https://github.com/winnerineast/TensorFlow-World)。 
- [pytorch](https://github.com/winnerineast/pytorch):大有赶超Tensorflow的脸书深度学习框架。
- [keras](https://github.com/winnerineast/keras)：在Theano宣布死亡之后，拥抱Tensorflow的最佳快速编程框架，非常适合于验证各种论文算法。
- [caffe](https://github.com/winnerineast/caffe)：这个世界上最强大，最难用，最广泛使用的深度学习框架，现在投奔Facebook，主推[caffe2](https://github.com/winnerineast/caffe2)。
- [caffe2](https://github.com/winnerineast/caffe2)：由于caffe框架的难用，所以Facebook开始打造第二个版本。
- [scikit-learn](https://github.com/winnerineast/scikit-learn)：大名鼎鼎的机器学习算法库。
- [hw](https://github.com/winnerineast/hw)：英伟达的深度学习加速器用到的一些库，用GPU的需要了解。
- [pyro](https://github.com/winnerineast/pyro)：一种通用概率模型算法框架，来自优步人工智能实验室。
- [deepo](https://github.com/winnerineast/deepo)：一个集成了 theano, [tensorflow](https://github.com/winnerineast/tensorflow), sonnet, pytorch, [keras](https://github.com/winnerineast/keras), lasagne, mxnet, cntk, [chainer](https://github.com/chainer/chainer), caffe, torch的容器，主要用于各种云计算资源部署。
- [chainer](https://github.com/chainer/chainer)：一种通用深度学习框架，主打视觉领域。
- [pytorch-qrnn](https://github.com/winnerineast/pytorch-qrnn)：Quasi-RNN的pytorch代码实现，这是一种LSTM的替代品，但是速度是英伟达的cuDNN极致优化后LSTM的2-17倍。
- [Paddle](https://github.com/winnerineast/Paddle):百度的深度学习框架。很中国。
- [Turicreate](https://github.com/winnerineast/turicreate)：苹果的通用深度学习框架，主打iOS应用。
- [predictionio](https://github.com/winnerineast/predictionio)：基于Spark，HBase和Spray的机器学习框架。
- [pattern](https://github.com/winnerineast/pattern)：多用途爬虫，NLP，机器学习，网络分析和可视化框架。
- [VisualDL](https://github.com/winnerineast/VisualDL)：可视化深度学习的过程框架。
- [TensorFlow-World-Resources](https://github.com/winnerineast/TensorFlow-World-Resources)：收集和整理了Tensorflow比较完整的系统，工具。
- [dm_control](https://github.com/winnerineast/dm_control)：Deep Mind的强化学习做的通用人工智能学习控制系统。
- [sonnet](https://github.com/winnerineast/sonnet)：基于Tensorflow的深度学习框架。
- [uTensor](https://github.com/winnerineast/uTensor)：主要为移动设备优化深度学习用，基于Tensorflow。
- [EvalAI](https://github.com/winnerineast/EvalAI)：一个深度学习机器学习评估框架。对标Kaggle。
- [aerosolve](https://github.com/winnerineast/aerosolve)：AirBnB的深度学习框架，据说对人界面友好。

## <a name="推荐系统类">推荐系统类</a>
- [librec](https://github.com/winnerineast/librec)：使用Java实现的完备的推荐系统，包含了从原始数据转化到训练数据集，相似度计算，训练，过滤，推荐的全过程。
- [predictionio](https://github.com/winnerineast/incubator-predictionio)：这是apache基于Spark和HBase的预测算法框架，适合于做推荐系统，相似度计算，分类算法。
- [tf_repos](https://github.com/winnerineast/tf_repos)：深度CTR算法，比较了Wide&Deep，NFM，AFM，FNN，PNN几种算法，使用Tensorflow框架。
- [Surprise](https://github.com/winnerineast/Surprise)：基于scikit的推荐系统。
- [TransNets](https://github.com/winnerineast/TransNets)：Learning to Transform for Recommendation的开源实现。
- [CaseRecommender](https://github.com/winnerineast/CaseRecommender)：包含了众多推荐算法。
- [DeepRecommender](https://github.com/winnerineast/DeepRecommender)：英伟达出品的深度学习做协同过滤。
- [triplet_recommendations_keras](https://github.com/winnerineast/triplet_recommendations_keras)：一种利用[LightFM](https://github.com/lyst/lightfm)实现的电影推荐算法。
- [RecommendationSystem](https://github.com/winnerineast/RecommendationSystem)：用Keras做的一系列推荐算法。

## <a name="深度学习电子书类">深度学习电子书类</a>
- [tensorflow-mnist-tutorial](https://github.com/winnerineast/tensorflow-mnist-tutorial)：这是学习Tensorflow的人必须看的东西。谷歌大神Martin Gorner在2017年python大会上讲演用的代码，比较好的诠释了Tensorflow怎么应用于深度学习的，使用的是MNIST数据集。
- [deeplearningbook-chinese](https://github.com/winnerineast/deeplearningbook-chinese):这是一本深度学习英文版的中文翻译，这本书中规中矩，适合于希望做神经网络底层优化的人系统学习，也能给应用神经网络的人一些有益的知识。
- [awesome-machine-learning](https://github.com/winnerineast/awesome-machine-learning)：综合型机器学习书和资源列表。
- [Visualizing-Convnets](https://github.com/winnerineast/Visualizing-Convnets)：可视化介绍卷机神经网络构建全过程。
- [python-parallel-programming-cookbook-cn](https://github.com/winnerineast/python-parallel-programming-cookbook-cn)：python并行计算编程。
- [App-DL](https://github.com/winnerineast/App-DL)：一个个人收集的各种深度学习资源列表。包含了初创公司列表。
- [deeplearning-mindmap](https://github.com/winnerineast/deeplearning-mindmap)和[machine-learning-mindmap](https://github.com/winnerineast/machine-learning-mindmap)是一对机器学习和深度学习知识概念图。
- [slambook](https://github.com/winnerineast/slambook)：一本自动驾驶的SLAM技术书。
- [Awesome-CoreML-Models](https://github.com/winnerineast/Awesome-CoreML-Models)：一个整理的比较好的机器学习和深度学习模型列表。

## ***********************************************************************************
## 以下内容和深度学习关系不大，但是工程化和落地深度学习技术，这些技术和工具都还是很有用处的。
## ***********************************************************************************

## <a name="数据集">数据集</a>
- [chinese-poetry](https://github.com/winnerineast/chinese-poetry):最全中华古诗词数据库, 唐宋两朝近一万四千古诗人, 接近5.5万首唐诗加26万宋诗. 两宋时期1564位词人，21050首词。
- [DuReader](https://github.com/winnerineast/DuReader)：百度的阅读理解训练数据集。
- [fashion-mnist](https://github.com/winnerineast/fashion-mnist)：一个类似MNIST的流行产品图片数据集。
- [youtube-8m](https://github.com/winnerineast/youtube-8m)：谷歌的youtube视频数据集。
- [boxscore-data](https://github.com/winnerineast/boxscore-data)：NBA数据集

## <a name="数据收集类">数据收集类</a>
- [scrapy](https://github.com/winnerineast/scrapy)：一个有名的python爬虫系统，可以结合一些Java Script库来爬动态页面。
- [ItChat](https://github.com/winnerineast/ItChat)：微信 python API，可以用来爬微信聊天记录，可以作为对话机器人的输出接口。
- [wxpy](https://github.com/winnerineast/wxpy)：微信机器人 / 可能是最优雅的微信个人号 API。
- [youtube-dl](https://github.com/winnerineast/youtube-dl)：下载视频数据的python工具。
- [labelme](https://github.com/winnerineast/labelme)：图片打标签的工具。
- [machinery](https://github.com/winnerineast/machinery)：一个处理摄像头视频的框架。

## <a name="数据格式化类">数据格式化类</a>
- [protobuf](https://github.com/winnerineast/protobuf)：谷歌的数据交换建模语言，方便python，java，c语言的数据交换。
- [FFmpeg](https://github.com/winnerineast/FFmpeg)：非常有名的视频，音频数据格式化和编解码器。

## <a name="数据存储类">数据存储类</a>
- [leveldb](https://github.com/winnerineast/leveldb):谷歌用来存储图片，文档的Key-Value数据库，特别适合于存放打标签的数据。

## <a name="数据可视化类">数据可视化类</a>
- [manim](https://github.com/winnerineast/manim)：在Latex文档里面做公式动画，比较适合于解释比较困难的公式。
- [facets](https://github.com/winnerineast/facets)：对于机器学习数据集的可视化工具。

## <a name="网络安全技术类">网络安全技术类</a>
- [AngelSword](https://github.com/winnerineast/AngelSword)：一个检测各类CMS漏洞的python代码，主要特色是支持很多CMS系统的接口渗透。

## <a name="其它">其它</a>
- [Celery](https://github.com/winnerineast/celery):非常有用的并行任务调度框架，适合于长时间数据处理。
- [flask](https://github.com/winnerineast/flask):非常方便的python网页服务框架。管理工具[Flower](https://github.com/winnerineast/flower)
- [dash](https://github.com/winnerineast/dash)：dash是python的响应式网页应用框架。
- [ray](https://github.com/winnerineast/ray)：python的分布式计算框架。
- [react](https://github.com/winnerineast/react)：毫无疑问，脸书希望打败手机原生应用的武器。
- [django](https://github.com/winnerineast/django)：可以用来做任何系统的快速开发原型，特别适合于人工智能系统。
