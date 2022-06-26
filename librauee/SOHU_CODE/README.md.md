## 运行所需包

* 参见requirements.txt

## 复现流程

* 将官方所给的初赛与复赛的数据集统一直接放到Sohu2022_data目录下（不要中间目录），其中用于推荐的recommend_content_entity.csv文件为了区分训练和测试添加了子目录
* data目录用来存储部分中间数据文件
* NLP部分   运行run_section1.sh即可进行训练预测，模型放在model目录下，生成section1.txt的提交文件
* 推荐部分   运行run_section2.sh即可进行训练预测，模型放在model目录下，生成section2.txt的提交文件  

## 相关中间数据与模型

* 后续通过云端打包的方式提供下载链接
* 主要包括NLP的模型，约为1.5G
* 测试集数据包含特征，约为1.5G
* NLP推理得到的情感特征约为0.1G