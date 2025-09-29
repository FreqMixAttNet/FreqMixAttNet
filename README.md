
---
>
> 🙋 Please let us know if you find out a mistake or have any suggestions!
> 
> 🌟 If you find this resource helpful, please consider to star this repository:

## Introduction
🏆 **FreqMixAttNet**, a novel crossdomain forecasting framework that unifies time and frequency representations via a domain-mixing attention mechanism.**.

🌟**** 



## Get Started

1. Install requirements. ```pip install -r requirements.txt```
    > If you are using **Python 3.8**, please change the `sktime` version in `requirements.txt` to `0.29.1`
2. Download data. You can download all datasets from [Google Driver](https://drive.google.com/u/0/uc?id=1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP&export=download), [Baidu Driver](https://pan.baidu.com/share/init?surl=r3KhGd0Q9PJIUZdfEYoymg&pwd=i9iy) or [Kaggle Datasets](https://www.kaggle.com/datasets/wentixiaogege/time-series-dataset). **All the datasets are well pre-processed** and can be used easily.
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the experiment results by:

```bash
bash ./scripts/etth1/etth1_96.sh
bash ......
```

## Main Results
We conduct extensive experiments to evaluate the performance and efficiency of FreqMixAttNet, covering long-term and short-term forecasting, including 18 real-world benchmarks and 15 baselines.
**🏆 FreqMixAttNet achieves consistent state-of-the-art performance in all benchmarks**, covering a large variety of series with different frequencies, variate numbers and real-world scenarios.

### Long-term Forecasting

To ensure fairness in model comparison, experiments were performed with standardized parameters, including aligned input lengths, batch sizes, and training epochs. Additionally, given that results in various studies often stem from hyperparameter optimization, we include outcomes from comprehensive parameter searches.

<p align="center">
<img src="./figures/long_results.png"  alt="" align=center />
</p>


## Model Abalations

To verify the effectiveness of each component of FreqMixAttNet, we provide the detailed ablation study on every possible design in both Past-Decomposable-Mixing and Future-Multipredictor-Mixing blocks on all 18 experiment benchmarks （see our paper for full results 😊）.

<p align="center">
<img src="./figures/ablation.png"  alt="" align=center />
</p>


## Further Reading

**Authors**: 

```bibtex


```

## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code and efforts.
- Time-Series-Library (https://github.com/thuml/Time-Series-Library)
- Autoformer (https://github.com/thuml/Autoformer)
- TimeMixer (https://github.com/kwuking/TimeMixer)

## Contact

If you have any questions or want to use the code, feel free to contact:

