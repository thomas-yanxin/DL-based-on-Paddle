# 常见的深度学习网络结构

## 全连接网络结构


<font face="楷体" size=4>

&emsp;&emsp;全连接(Fully Connected, FC)网络结构是**最基本的神经网络/深度神经网络层**，全连接层的每一个节点都与上一层的所有节点相连。全连接层在早期主要用于对提取的特征进行**分类**，然而由于全连接层所有的输出与输入都是相连的，一般全连接层的参数是最多的，这需要相当数量的**存储空间和计算空间**。参数的冗余问题使单纯的FC组成的常规神经网络很少会被应用于较为复杂的场景中。常规神经网络一般用于依赖所有特征的简单场景，比如房价预测模型和在线广告推荐模型使用的都是相对标准的全连接神经网络。
    <div align=center><img width="500" height="450" src="https://ai-studio-static-online.cdn.bcebos.com/cd5e82cbe7e14140832ce0c9e556710e2db800b04ef84d9e92bd7cb51e2fb2b8"/></div>   

  
  
</font>

## 卷积神经网络
<font face="楷体" size=4>
 
  
&emsp;&emsp;卷积神经网络(CNN)是一种专门用来处理**具有类似网络结构的数据的神经网络**，如图像数据（可以看作二维的像素网格）。与FC不同的地方在于，CNN的上下层神经元并不能都直接连接，而是通过 **“卷积核”** 作为中介，通过“核”的共享大大减少了隐藏层的参数。简单的CNN是一系列层，并且每个层都通过一个可微函数将一个量转化为另一个量，这些层主要包括**卷积层**( Convolutional Layer )、**池化层**( Pooling Layer )和全连接层( FC Layer )。卷积网络在诸多应用领域，尤其是大型图像处理的场景都取得了很好的应用效果。  
  &emsp;&emsp;如图展示了CNN的结构形式，一个神经元以三位排列组成卷积神经网路 （宽度、高度、深度），如其中一个层所展示的，CNN的每一层都将3D的输入量转化为3D的输出量。
  <div align=center><img width="500" height="450" src="https://ai-studio-static-online.cdn.bcebos.com/4521e9ac7d264feaa6821352b037bb12ed62ad0b1f744ac48f45661817ba6d27"/></div>


  
  
 </font>

## 循环神经网络

<font face="楷体" size=4>
  &emsp;&emsp;循环神经网络(RNN)也是常用的深度学习模型之一，就如CNN是专门用于处理网络化数据（例如图像）的神经网络，RNN是一种 **专门处理序列数据的神经网络** 。如音频中含有时间成分，因此音频可以被表示为一维时间序列；语言中的单词都是逐个出现的，因此语言的表示方式也是序列数据。RNN在机器翻译、语音识别等领域均有非常好的表现。
    <div align=center><img width="500" height="450" src="https://ai-studio-static-online.cdn.bcebos.com/c4ea704742aa4f52b6bbc4d4030009307597026e746f4954b6f273cc5b23c7ee"/></div>
  
   </font>
