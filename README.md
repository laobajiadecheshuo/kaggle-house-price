# Kaggle 房价预测（House Prices）

《动手学深度学习》（d2l-zh）第 4 章实战：[House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)。
任务：根据 Ames 房产的 79 个特征预测售价，官方指标为 **log RMSE**。

## 方法与结果

流程：数值特征标准化 → 缺失值填 0 → 类别特征独热编码（79 → 330 维）→ 线性回归 + Adam → 5 折交叉验证调参。

| 配置 | 5 折验证 log RMSE |
|------|------------------|
| 教材默认（lr=5, 100 epoch） | — |
| 手调（lr=2.0, wd=0.01, 500 epoch） | 0.1480 |
| **网格搜索最优（lr=3.0, wd=0, 500 epoch）** | **0.1469** |

调参过程：`search_linear2.py` 网格搜索 440 组配置（lr × weight_decay × epochs × batch_size），
结果记录在 `search_linear2_results.txt`；`autogluon_house.py` 为 AutoGluon 自动化对照。

## 文件说明

```
├── kaggle-house-price.ipynb     # 主 notebook（d2l 实战 + 踩坑修复 + 调优后超参）
├── search_hp.py                 # 超参搜索脚本（版本 1）
├── search_linear.py             # 超参搜索脚本（版本 2）
├── search_linear2.py            # 超参搜索脚本（版本 3，440 组网格）
├── search_linear2_results.txt   # 搜索日志
├── autogluon_house.py           # AutoGluon 对照
└── submission.csv / submission_autogluon.csv   # Kaggle 提交文件
```

notebook 里的踩坑修复：新版 pandas 下 `.values` 需要 `.astype(float)`
（object dtype 直接转 tensor 会报错）。

## 复现

数据经 d2l 的 `DATA_HUB` 自动下载到 `../data`（sha1 校验缓存），克隆后直接运行 notebook 即可。

## 致谢

[notebook](https://github.com/d2l-ai/d2l-zh) 结构来自《动手学深度学习》4.10 节，超参与修复为本项目工作。
