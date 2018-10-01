# DoodleRecognition

## Model
目前借鉴别的组的Kernel，采用 *ResNet18* + *CyclicLR* 训练我们的模型 日后可能会加入 *RNN* 提升模型的准确度
如果DeepLearning的知识点需要恶补，参见 [DeepLearningAI](https://mooc.study.163.com/smartSpec/detail/1001319001.htm)中 `01` `02` `04` `05`章节
如果Pytorch不熟悉，参见 [Pytorch tutorials](https://pytorch.org/tutorials/) 一天时间速成 Pytorch

------

## 环境
1. conda(pandas最好用0.23.4 其他版本读取CSV数据老是出错)
```
torchvision               0.2.1                    
scikit-learn              0.19.1         
scipy                     1.1.0           
pytorch                   0.4.1         
python                    3.6.6                
pandas                    0.23.4          
opencv-python             3.4.3.18                 
numpy                     1.15.1          
numpy-base                1.15.1         
matplotlib                2.2.3            
```

2. Matplotlib
```
import matplotlib
matplotlib.use('Agg')
```

如果要用远程的服务器绘制图像，需要用XForward11 参见[在pycharm中使用远程的matplotlib](https://stackoverflow.com/questions/30813370/how-can-i-enable-x-11-forwarding-in-pycharm-when-connecting-to-vagrant-or-a-rem/32945380#32945380)

如果只是在本地的服务器绘制图像，将 *matplotlib.use('Agg')* 注释掉即可 系统默认用 *TkAgg*

----

## Results
```
Params to learn:
	 module.pretrained_model.conv1.weight
	 module.pretrained_model.bn1.weight
	 module.pretrained_model.bn1.bias
	 module.pretrained_model.layer1.0.conv1.weight
	 module.pretrained_model.layer1.0.bn1.weight
	 module.pretrained_model.layer1.0.bn1.bias
	 module.pretrained_model.layer1.0.conv2.weight
	 module.pretrained_model.layer1.0.bn2.weight
	 module.pretrained_model.layer1.0.bn2.bias
	 module.pretrained_model.layer1.1.conv1.weight
	 module.pretrained_model.layer1.1.bn1.weight
	 module.pretrained_model.layer1.1.bn1.bias
	 module.pretrained_model.layer1.1.conv2.weight
	 module.pretrained_model.layer1.1.bn2.weight
	 module.pretrained_model.layer1.1.bn2.bias
	 module.pretrained_model.layer2.0.conv1.weight
	 module.pretrained_model.layer2.0.bn1.weight
	 module.pretrained_model.layer2.0.bn1.bias
	 module.pretrained_model.layer2.0.conv2.weight
	 module.pretrained_model.layer2.0.bn2.weight
	 module.pretrained_model.layer2.0.bn2.bias
	 module.pretrained_model.layer2.0.downsample.0.weight
	 module.pretrained_model.layer2.0.downsample.1.weight
	 module.pretrained_model.layer2.0.downsample.1.bias
	 module.pretrained_model.layer2.1.conv1.weight
	 module.pretrained_model.layer2.1.bn1.weight
	 module.pretrained_model.layer2.1.bn1.bias
	 module.pretrained_model.layer2.1.conv2.weight
	 module.pretrained_model.layer2.1.bn2.weight
	 module.pretrained_model.layer2.1.bn2.bias
	 module.pretrained_model.layer3.0.conv1.weight
	 module.pretrained_model.layer3.0.bn1.weight
	 module.pretrained_model.layer3.0.bn1.bias
	 module.pretrained_model.layer3.0.conv2.weight
	 module.pretrained_model.layer3.0.bn2.weight
	 module.pretrained_model.layer3.0.bn2.bias
	 module.pretrained_model.layer3.0.downsample.0.weight
	 module.pretrained_model.layer3.0.downsample.1.weight
	 module.pretrained_model.layer3.0.downsample.1.bias
	 module.pretrained_model.layer3.1.conv1.weight
	 module.pretrained_model.layer3.1.bn1.weight
	 module.pretrained_model.layer3.1.bn1.bias
	 module.pretrained_model.layer3.1.conv2.weight
	 module.pretrained_model.layer3.1.bn2.weight
	 module.pretrained_model.layer3.1.bn2.bias
	 module.pretrained_model.layer4.0.conv1.weight
	 module.pretrained_model.layer4.0.bn1.weight
	 module.pretrained_model.layer4.0.bn1.bias
	 module.pretrained_model.layer4.0.conv2.weight
	 module.pretrained_model.layer4.0.bn2.weight
	 module.pretrained_model.layer4.0.bn2.bias
	 module.pretrained_model.layer4.0.downsample.0.weight
	 module.pretrained_model.layer4.0.downsample.1.weight
	 module.pretrained_model.layer4.0.downsample.1.bias
	 module.pretrained_model.layer4.1.conv1.weight
	 module.pretrained_model.layer4.1.bn1.weight
	 module.pretrained_model.layer4.1.bn1.bias
	 module.pretrained_model.layer4.1.conv2.weight
	 module.pretrained_model.layer4.1.bn2.weight
	 module.pretrained_model.layer4.1.bn2.bias
	 module.pretrained_model.fc.weight
	 module.pretrained_model.fc.bias
	 module.linear.weight
	 module.linear.bias
MIXUP
Epoch 0/24
----------
index: 0  running_loss 1.4031  running_corrects 0.3125
index: 1  running_loss 1.4532  running_corrects 0.2812
index: 2  running_loss 1.4397  running_corrects 0.2604
index: 3  running_loss 1.4200  running_corrects 0.2773
index: 4  running_loss 1.4116  running_corrects 0.2625
index: 5  running_loss 1.3817  running_corrects 0.2943
index: 6  running_loss 1.3536  running_corrects 0.3326
index: 7  running_loss 1.3272  running_corrects 0.3750
index: 8  running_loss 1.3057  running_corrects 0.3993
index: 9  running_loss 1.2794  running_corrects 0.4266
index: 10  running_loss 1.2558  running_corrects 0.4503
index: 11  running_loss 1.2339  running_corrects 0.4701
index: 12  running_loss 1.2139  running_corrects 0.4820
index: 13  running_loss 1.1840  running_corrects 0.5022
index: 14  running_loss 1.1608  running_corrects 0.5208
index: 15  running_loss 1.1358  running_corrects 0.5430
index: 16  running_loss 1.1146  running_corrects 0.5579
index: 17  running_loss 1.0882  running_corrects 0.5764
index: 18  running_loss 1.0663  running_corrects 0.5896
index: 19  running_loss 1.0437  running_corrects 0.6016
index: 20  running_loss 1.0200  running_corrects 0.6168
index: 21  running_loss 0.9997  running_corrects 0.6300
index: 22  running_loss 0.9828  running_corrects 0.6399
index: 23  running_loss 0.9615  running_corrects 0.6517
index: 24  running_loss 0.9439  running_corrects 0.6619
index: 25  running_loss 0.9275  running_corrects 0.6707
index: 26  running_loss 0.9111  running_corrects 0.6794
index: 27  running_loss 0.8977  running_corrects 0.6853
index: 28  running_loss 0.8870  running_corrects 0.6880
index: 29  running_loss 0.8714  running_corrects 0.6958
index: 30  running_loss 0.8527  running_corrects 0.7046
index: 31  running_loss 0.8416  running_corrects 0.7100
index: 32  running_loss 0.8243  running_corrects 0.7169
index: 33  running_loss 0.8124  running_corrects 0.7229
index: 34  running_loss 0.8015  running_corrects 0.7259
index: 35  running_loss 0.7895  running_corrects 0.7300
index: 36  running_loss 0.7772  running_corrects 0.7352
index: 37  running_loss 0.7645  running_corrects 0.7397
index: 38  running_loss 0.7537  running_corrects 0.7444
index: 39  running_loss 0.7451  running_corrects 0.7484
index: 40  running_loss 0.7342  running_corrects 0.7523
index: 41  running_loss 0.7234  running_corrects 0.7571
index: 42  running_loss 0.7143  running_corrects 0.7605
index: 43  running_loss 0.7052  running_corrects 0.7646
index: 44  running_loss 0.6963  running_corrects 0.7684
index: 45  running_loss 0.6880  running_corrects 0.7724
index: 46  running_loss 0.6796  running_corrects 0.7753
index: 47  running_loss 0.6720  running_corrects 0.7790
index: 48  running_loss 0.6641  running_corrects 0.7809
index: 49  running_loss 0.6555  running_corrects 0.7847
index: 50  running_loss 0.6497  running_corrects 0.7868
index: 51  running_loss 0.6416  running_corrects 0.7897
index: 52  running_loss 0.6361  running_corrects 0.7919
index: 53  running_loss 0.6295  running_corrects 0.7943
index: 54  running_loss 0.6212  running_corrects 0.7974
index: 55  running_loss 0.6141  running_corrects 0.7997
index: 56  running_loss 0.6072  running_corrects 0.8024
index: 57  running_loss 0.5993  running_corrects 0.8055
index: 58  running_loss 0.5942  running_corrects 0.8072
index: 59  running_loss 0.5876  running_corrects 0.8096
index: 60  running_loss 0.5809  running_corrects 0.8117
index: 61  running_loss 0.5744  running_corrects 0.8143
train Loss: 0.5698 Acc: 0.8077
val Loss: 0.2214 Acc: 0.8753

Epoch 1/24
----------
index: 0  running_loss 0.1710  running_corrects 0.9375
index: 1  running_loss 0.1657  running_corrects 0.9531
index: 2  running_loss 0.1515  running_corrects 0.9583
index: 3  running_loss 0.1742  running_corrects 0.9492
index: 4  running_loss 0.1950  running_corrects 0.9469
index: 5  running_loss 0.2064  running_corrects 0.9427
```

----

## 资料
1. [QuickDraw DataSet](https://github.com/googlecreativelab/quickdraw-dataset#the-raw-moderated-dataset)
2. [Pytorch Tutorial -- Train a Image Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py)
3. [FineTuning TorchVision Models](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)
4. [ResNet, CLR, Pytorch](https://www.kaggle.com/prajjwal/resnet-clr-pytorch)
5. [CNN LSTM Model](https://www.kaggle.com/kmader/quickdraw-baseline-lstm-reading-and-submission)

----

## TO DO LIST
1. 数据量实在太大了 350个不同的种类 每个种类十几万条数据
2. 试着**借鉴**其他Kernel的模型 比如上面那个 CNN LSTM Model
