# 2022搜狐校园 情感分析 × 推荐排序 算法大赛

## 曝光特征

* 用户文章设备等ID类特征的全局曝光次数和当天曝光次数

## 交叉特征

* 统计用户ID与所有文章侧ID
* 文章ID与所有用户侧ID的类别交叉

## 差分特征

* 用户、文章等ID类特征的曝光时间差聚合统计

## Word2vec

利用历史文章浏览序列构造embedding特征

## Target Attention

* 利用序列embedding与当前文章做attention（协方差）
* 做统计特征

## 实体情感

* 统计特征

## Position Bias

* 排序靠前的点击率较高
* 曝光顺序影响label分布
* rank排序

**参考**：https://github.com/librauee