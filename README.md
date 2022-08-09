# Kmeans

数据工程基础实践——Kmeas

## Kmeans算法概述
  Kmeans算法是常用的无监督学习，将数据集聚类为给定簇数K类。
Kmeans采用直接分配的方法寻找指定个数簇各自的形心。初始时，选定簇的个数K，选定每个簇的形心位置μK 。对于一点P,它属于与μK 欧式距离最小的那个簇RK。接着对所有属于RK 中的所有的点坐标累计求均值作为新的的μK ，然后重复前两步直至μK 不再改变算法结束，聚类完成。

### Kmeans(含优化)算法流程图
![image](https://user-images.githubusercontent.com/92938836/183614160-9a571067-c925-4379-8e92-79c9c9cbf369.png)

### Kmeans算法核心步骤代码实现
#### 欧式距离函数
<img width="640" alt="image" src="https://user-images.githubusercontent.com/92938836/183614252-980504ca-dec1-477b-a96b-9bc48b71baaf.png">

欧式距离用于计算点P与簇心的距离，这里采用np.square一元函数可以提高计算速度。
#### Kmeans算法核心步骤实现
![image](https://user-images.githubusercontent.com/92938836/183614271-c1b94a30-eba5-4e16-baa1-f62b82c0fe12.png)

这里我将KMeans算法用Class封装，便于使用，在理解的算法的本质后我们只要按照流程图书写对应的代码即可。我使用一个多维向量R[i]来表示Xi的标签，那么在第一个循环中我们使用R[n,np.argmin(distance)]=1就可以将距离最近的那个位标记位1。使用多维向量做标签的好处在于我们更新Mu(簇心向量)的时候可以直接累积求和求平均值更新。在更新mu的过程中，np.sum(R[:,k]*X[:,j])计算了属于簇Ri中所有距离之和，因为标签为0或1，np.sum(R[:,k])又统计了个数。二者相除即可求出均值来确定新的簇心。 

### Kmeans的优缺点

Kmeans的优点在于：

原理简单易于理解，收敛速度快，同时Kmeans的聚类效果优秀。

Kmeans的缺点在于：

Kmeans根据欧式距离进行划分，对噪声和孤立的点很敏感可能由于某些数据导致聚类效果不佳。

Kmeans初始化K的簇数并不好确定。

同样由于利用欧氏距离，初始簇心的选择的不同对最终结果会产生影响。有甚者可能由于初始簇心的不同而导致最终结果截然不同，算法失去意义。

Kmeans的算法时间复杂度与样本数量成线性关系，由于每次计算簇心或者R都要对所有质心重新计算，总体的聚类时间开销大。

### 缺点对应策略

#### Ⅰ.簇数K的确定 

K值的选择会对kmeans的结果产生重大影响。我们往往并不知道数据集实际上共有几类，那么如何找到一个最接近真实值的K值呢？我们这里先引入一个SSE的概念。
      
> SSE，sum of the squared errors，误差的平方和。在K-means 算法中，SSE 计算的是每类中心点与其同类成员距离的平方和。
      
不难理解随着聚类数k的增大，样本划分会更加精细，每个簇的聚合程度会逐渐提高，那么误差平方和SSE自然会逐渐变小。当k小于最佳聚类数时，k的增大会大幅增加每个簇的聚合程度，故SSE的下降幅度会很大；当k到达最佳聚类数时，再增加k所得到的聚合程度，回报会迅速变小，所以SSE的下降幅度会骤减，然后随着k值的继续增大而趋于平缓。也就是说SSE和 k 的关系图是一个手肘的形状，而这个肘部对应的k值就是数据的最佳聚类数。这也是该方法被称为手肘法的原因。

当然手肘法并不是万能，因为有些时候SSE和K的关系并不是一个线性的关系，并没有一个曲率很高的k值，这个时候我们就需要另寻他法。K值得选择并没有绝对的正确或者错误，我们并不可能找到一个通用的计算法来获得所有数据集的真实的K值，亦或者数据集原本就没有明确的几类。

**SEE的python实现**
<img width="640" alt="image" src="https://user-images.githubusercontent.com/92938836/183615668-5baada37-c704-4762-8b1b-6d6baec1fc4e.png">

**实用案例Test**
<img width="436" alt="image" src="https://user-images.githubusercontent.com/92938836/183615839-fcbafb27-1ed3-4eeb-b389-c0832fddc293.png">
可以看到这里最接近真实值的K值出现在3处，而在3以后的点回报并不高。

#### Ⅱ 初始质心的选择 Kmeans++优化

在初始化的过程中，初始簇心的选择也非常重要，不同的初始簇心可能导致不同的实验结果，我们常用的选择方法有：随机选择、层次化聚类、kmeans++。这里我们采用kmea++来初始化簇心。

Kmeans++的算法思想简而言之就是仅可能选择欧式距离相对远的作为初始化质心。这样做的原因我认为是相似的样本在空间中分布也相对聚集，换言之说不相似的点相对距离就远。那么我们只要尽可能的选择距离相对远的点就可以让初始点接近每个簇的最终簇心位置就可以避免随机选择的问题。

这样做的好处有两点：一是可以实验效果更好，二是因为初始化簇心接近最终簇心所以总体的计算复杂度会下降。

接下来给出Kmeans++的思路流程图：
![image](https://user-images.githubusercontent.com/92938836/183616362-761dc9f2-94bd-48b6-a64a-2dd179e18853.png)

#### Kmeans++的代码实现

**Min_Distance函数**

<img width="640" alt="image" src="https://user-images.githubusercontent.com/92938836/183616545-c7095a50-a8c3-4e29-a8e4-500b136e5e5e.png">

**Kmeans++函数**

<img width="640" alt="image" src="https://user-images.githubusercontent.com/92938836/183616639-5150a4fa-66e2-4b4f-a658-9bb3050407a3.png">

这里我采用了先初始化一个零向量，然后将计算所得到的新簇心通过np.argmax()函数来找到距离最远的簇心然后用压栈的方式压入簇心组。这里可能会产生一个疑问原簇心是否会被再次压入簇心组，由于点到自身的距离为0所以并不会产生这样的问题。相反如果在遍历的时候进行判断会导致额外的时间开销，如果在源数据集上做改动又会导致对数据集的破坏，当然可以clone一个数据集去掉被选中的点用来做kmeans++。

### 降维与数据可视化

为了更直观的观察实验结果我们需要进行数据可视化，对于二维数据我们可以很直观的将数据集直接在平面中展示出来，但实验所用的数据集往往是高纬度的并不可以直接观察，所以我们需要把高纬度的数据降维至二维/三维数据进行观察。由于对降维算法知识基础不够理解，这里直接使用tsne/pca库进行降维可视化。

```python
# 导入TSNE降维工具 
from sklearn.manifold import TSNE
#TSNE使用
tsne = TSNE() 构造一个TSNE对象
test1.X = tsne.fit_transform(test1.X) #调用transform方法对数据集降维更新数据集
# 导入PCA 降维工具
from sklearn.decomposition import PCA
# PCA使用
pca = PCA(n_components=2) #指定降维维数，构造一个pca对象
self.X = pca.fit_transform(self.X) # 调用transform方法对数据集降维更新数据集
```

**绘图函数实现**

<img width="514" alt="image" src="https://user-images.githubusercontent.com/92938836/183617070-f0d2f7bc-12a1-496d-829b-af7dd83ad72e.png">

首先要确定的就是范围，展示要将所有的点都包括在内并且不可以因为范围过大或者过小而导致数据分布过于集中失去观察价值，为了实现这一点只需要将范围设置为最小与最大的倍数即可。接着我们要再一次计算中心点的位置，因为我们是先对数据进行Kmeans聚类，再降维可视化。当我们对数据降维完成后，簇心向量并没有被降维，展示在图中的位置会错位。 接着我们只要选出数据集中属于第k类点的横纵坐标作为plt.plot的参数就可以完成绘制。

**实用案例：PCA降维**

<img width="436" alt="image" src="https://user-images.githubusercontent.com/92938836/183617211-407f02fb-ea38-43b1-aca6-d078117485a4.png">

<img width="436" alt="image" src="https://user-images.githubusercontent.com/92938836/183617282-c9e2709a-dbba-47d9-9d7c-e16567d21931.png">

**实用案例 TSNE降维**

<img width="436" alt="image" src="https://user-images.githubusercontent.com/92938836/183617340-7a610d07-0748-46a7-b02d-9cad06866b3d.png">

<img width="436" alt="image" src="https://user-images.githubusercontent.com/92938836/183617358-ba3a225d-1a8b-46d1-babf-55ccc2c2f0b4.png">

通过案例不难看出TSNE可以更好的展示聚集的效果，因为TSNE降维是按照分布函数进行降维，越相似的点越会被分布在相近的地方自然会如此。

### 数据集测试

**所用到的库**

<img width="640" alt="image" src="https://user-images.githubusercontent.com/92938836/183617459-f414f64b-d0c4-494a-bc9a-71cd60aedbb3.png">

#### 1.IRIS 数据集
测试用例函数
由于sklearn内自带UCI数据集，直接导入使用即可

<img width="640" alt="image" src="https://user-images.githubusercontent.com/92938836/183617599-ebdf6b45-d4d8-4669-a00e-53676c254743.png">

#### 2.WINE 数据集

测试用例函数

<img width="640" alt="image" src="https://user-images.githubusercontent.com/92938836/183617717-e046512b-8337-4070-a1b6-c09d466adf6c.png">

#### 3.WINE_QUALITY_WHITE 数据集

测试用例函数

<img width="625" alt="image" src="https://user-images.githubusercontent.com/92938836/183617826-28ef704d-68ac-4add-97ba-938e904f013e.png">

#### 4. WINE_QUALITY_RED 数据集

测试用例函数

![image](https://user-images.githubusercontent.com/92938836/183617928-a69021d3-24bb-42a9-a08a-951e0b0919be.png)

![image](https://user-images.githubusercontent.com/92938836/183617953-55ce2d6c-093a-4060-ac2d-c735abdf3cc3.png)
