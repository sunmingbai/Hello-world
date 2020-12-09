基于梯度树提升的回归模型

线性模型虽然具有简单、易于解释的特点，但拟合与预测能力可能较弱，故本项目选择了集成的机器学习方法进行建模。

a)**基学习器选择**：综合考虑模型解释性和后续集成难度，选择决策树作为基学习器。
b)**集成：梯度提升决策树(GBDT)**：通过串行的方式连接各基学习器，相互依赖层层叠加，每一个新的学习器在训练的时候，针对前一个基学习器未能完全拟合的残差继续训练，最终所有弱分类器的结果相加等于预测值。
c) **优化：XGBoost/lightGBM**：XGBoost本质上还是一个GBDT，但是通过设计使得速度和效率大幅提高（加入正则惩罚，防止过拟合；每轮训练可以对样本采样；对缺失值可以处理）；LightGBM也是实现GBDT的框架，支持高效率的并行训练（速度比xgb更快，内存占用更小；采用Histogram的决策树算法等）



参考资料：

1. [XBG](：https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/3.3%20XGBoost/3.3%20XGBoost.md)

2. [LGB](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/3.4%20LightGBM/3.4%20LightGBM.md)