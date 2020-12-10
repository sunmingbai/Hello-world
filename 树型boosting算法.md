## 分类

集成学习根据各个弱分类器之间有无依赖关系，分为Boosting和Bagging两大流派：

1. Boosting流派，各分类器之间有依赖关系，必须串行，比如Adaboost、GBDT(Gradient Boosting Decision Tree)、Xgboost
2. Bagging流派，各分类器之间没有依赖关系，可各自并行，比如随机森林（Random Forest）



## AdaBoost

不断改变权重的算法(不是树)



# 树型算法：



## CART

![img](../../Typora/v2-68dcbdc8949d606955f0bc2acd24614e_1440w.jpg)



通过一些判断条件将空间切割成若干个子空间$(R_1,R_2,...R_M)$，每个子空间$R_m$用$c_m$来作为预测值（回归值）预测Y：
$$
\hat{f}(X)=\sum_{m=1}^Mc_mI\{X\in R_m\}
$$
如果我们采用最小化平方和$\sum(y_i-f(x_i))^2$的方式预测，可以证明最优的$\hat{c}_m$是区域中$y_i$的均值：
$$
\hat{c}_m=ave(y_i|x_i\in R_m)
$$






参考资料：

1. https://zhuanlan.zhihu.com/p/82054400



## BDT



## GBDT



## xgboost

### 核心思想

核心思想是对GBDT（Gradient Boosting Decision Tree) 一样，不断生产新的树，对前面的树未能拟合的部分进行补充优化。
$$
\hat{y} = \phi(x_i) = \sum_{k=1}^{K}f_k(x_i) \\
where \; F = \{f(x)=w_q(x)\}\quad \\
$$
$Note：w_q(x)是叶子节点q的分数$，每次添加新的树分类器的原则是，满足目标函数最小化（包括误差项、正则惩罚项）。

### 具体步骤

1. 不断生成新的树（选择新的特征进行切分），每次添加一个树，其实是学习一个新函数，去拟合上次预测的残差；
2. 当我们训练完成得到k棵树，我们要预测一个样本的分数，其实就是根据这个样本的特征，在每棵树中会落到对应的一个叶子节点，每个叶子节点就对应一个分数。
3. 最后只需要将每棵树对应的分数加起来就是该样本的预测值。

4. 目标函数（$i$是训练样本，k是分类器）

$$
L(\phi) = \sum_i l(\hat{y}_i,y_i)+\sum_k \Omega(f_k)\label{2}
$$

第一部分是损失函数（**衡量误差**），第二部分是正则化项（**衡量复杂度**）

对于每一个树（分类函数）：$\Omega(f)=\gamma T+\frac12\lambda||w||^2$，T表示叶子节点的个数，w表示叶子节点的分数，通过$l_2$范数限制（节点数量多，有过拟合的风险，单个叶子节点分数大有极端值的风险）。

其中$\hat{y}_i$通过不断添加树来逼近真实值的，这个过程可以表示如下：
$$
\hat{y}_i^{(t)}=\hat{y}_i^{(t-1)}+f_t(x_i)
$$
$\hat{y}_i^{(t-1)}$表示经过$t-1$个分类器后的预测值（累计和），$f_t(x_i)$则是第t个分类器的叶子分数。



### 目标函数和损失函数

公式$\eqref{2}$的最小化，实际上是在选择$f_t(x)$分类器，根据惩罚函数的不同：

- 均方误差损失函数：$l(y_i,\hat{y_i}) = (y_i-\hat{y}_i)^2$

目标函数可以改写为：
$$
\begin{equation}
\begin{aligned}
obj^{(t)} &= \sum_i l(\hat{y}_i^{(t)},y_i)+\sum_t \Omega(f_t) \\
&= \sum_i [(\hat{y}_i^{(t-1)}+f_t(x_i)-y_i)^2]+\Omega(f_t)+CONST\\
&= \sum_i [(\hat{y}_i^{(t-1)}-y_i+f_t(x_i))^2]+\Omega(f_t)+CONST\\
&=\sum_i [(\hat{y}_i^{(t-1)}-y_i)^2+2(\hat{y}_i^{(t-1)}-y_i)f_t(x_i) +f_t(x_i)^2]+\Omega(f_t)+CONST\\
&=\sum_i [l(\hat{y}_i^{(t-1)},y_i)+2(\hat{y}_i^{(t-1)}-y_i)f_t(x_i) +f_t(x_i)^2]+\Omega(f_t)+CONST

\label{4}
\end{aligned}
\end{equation}
$$

- 如果不是均方误差函数，则可以利用泰勒公式展开：

$$
f(x+\Delta x) \approx  f(x)+f'(x)\Delta x+\frac12f''(x)\Delta x^2
$$

定义（类似梯度下降和牛顿法）：
$$
g_i=\partial_{\hat{y}^{(t-1)}}l(y_i,\hat{y}_i^{(t-1)}) \\
h_i = \partial^2_{\hat{y}^{(t-1)}}l(y_i,\hat{y}_i^{(t-1)}) \\
$$
目标函数（第t-1和第t次添加树后的转换关系）
$$
obj^{(t)}\approx\sum_{i=1}^n [l(y_i,\hat{y}_i^{(t-1)})+g_if_t(x_i)+\frac12h_if_t^2(x_i)]+\Omega(f_t)+CONST
$$


### 目标函数化简式

这个目标函数中，$l(y_i,\hat{y}_i^{(t-1)})$是根据上一次添加树是已知的，$g_i,h_i$已知，可变的只有$f_t$了，去掉常数项可以得到下面的最简的式子：
$$
\begin{equation}
\begin{aligned}
obj^{(t)}&\approx \sum_{i=1}^n [g_if_t(x_i)+\frac12h_if_t^2(x_i)]+\Omega(f_t)\\
&= \sum_{i=1}^n[g_iw_q(x_i)+\frac12h_iw_q^2(x_i)]+\gamma T+\frac12\lambda \sum_{j=1}^Tw_j^2\\
&= \sum_{i=1}^n[(\sum\nolimits_{i\in I_j} g_i)w_j +\frac12 (\sum\nolimits_{i\in I_j}h_i+\lambda)w_j^2]+\gamma T\\
\end{aligned}
\end{equation}
$$
$Note$：**其中$I_j$被定义为每个叶节点$j$上面样本下标的集合$I_j=\{i|q(x_i)=j\}$**，g是一阶导数，h是二阶导数

这一步是由于xgboost目标函数第二部分加了两个正则项，一个是叶子节点个数(T),一个是叶子节点的分数(w)。这个定义里的$q(x_i)$要表达的是：每个样本值$x_i$ 都能通过函数$q(x_i)$映射到树上的某个叶子节点。

定义：
$$
G_j=\sum_{i\in I_j} g_i \quad H_j=\sum_{i\in I_j}h_i
$$
目标函数可化简为：
$$
\begin{equation}
\begin{aligned}
Obj^{(t)} &= \sum_{j=1}^T[(\sum\nolimits_{i\in I_j}g_i)w_j+\frac12(\sum\nolimits_{i\in I_j }h_i+\lambda)w_j^2]+\gamma T\\
&=\sum_{j=1}^T[G_jw_j+\frac12(H_j+\lambda)w_j^2]+\gamma T
\end{aligned}
\end{equation}
$$
通过对$w_j$求导，得到：
$$
w_j^*=-\frac{G_j}{H_j+\lambda}
$$
将最优解带入：
$$
Obj = -\frac12\sum_{j=1}^T \frac{G_j^2}{H_j+\lambda}+\gamma T
$$
Obj代表了当我们指定一个树的结构的时候，我们在目标上面最多减少多少。我们可以把它叫做结构分数(structure score)。这个数越小（绝对值越大），说明这个树的结构越好。



### 节点分裂

我们从头到尾了解了xgboost如何优化、如何计算，但树到底长啥样，我们却一直没看到。很显然，一棵树的生成是由一个节点一分为二，然后不断分裂最终形成为整棵树。那么树怎么分裂的就成为了接下来我们要探讨的关键。对于一个叶子节点如何进行分裂，xgboost作者在其原始论文中给出了两种分裂节点的方法。

1. **枚举所有不同树结构的贪心法**

现在的情况是只要知道树的结构，就能得到一个该结构下的最好分数，那如何确定树的结构呢？

一个想当然的方法是：不断地枚举不同树的结构，然后利用打分函数来寻找出一个最优结构的树，接着加入到模型中，不断重复这样的操作。而再一想，你会意识到要枚举的状态太多了，基本属于无穷种，那咋办呢？

我们试下贪心法，从树深度0开始，每一节点都遍历所有的特征，比如年龄、性别等等，然后对于某个特征，先按照该特征里的值进行排序，然后线性扫描该特征进而确定最好的分割点，最后对所有特征进行分割后，我们选择所谓的增益Gain最高的那个特征，而Gain如何计算呢？

<img src="../../Typora/quesbase64153438710772062978.png" alt="img" style="zoom:70%;" />

2. **近似算法**

总结



<img src="../../Typora/quesbase64153438717157194254.png" alt="img" style="zoom:75%;" />







参考资料

1. [通俗理解kaggle比赛大杀器xgboost](https://blog.csdn.net/v_JULY_v/article/details/81410574)
2. [机器学习算法之XGBoost](https://www.biaodianfu.com/xgboost.html)
3. https://www.julyedu.com/question/big/kp_id/23/ques_id/2590



## lightGBM



参考资料

1. [机器学习算法之LightGBM](https://www.biaodianfu.com/lightgbm.html)