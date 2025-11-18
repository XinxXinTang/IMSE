<!-- # MUSE: Flexible Voiceprint Receptive Fields and Multi-Path Fusion Enhanced Taylor Transformer for U-Net-based Speech Enhancemen (Accepeted at Interspeech 2024)
### Zizhen Lin, Xiaoting Chen, Junyu Wang

**Abstract:** 
 Achieving a balance between lightweight design and high performance remains a challenging task for speech enhancement. In this paper, we introduce Multi-path Enhanced Taylor (MET) Transformer based U-net for Speech Enhancement (MUSE), a lightweight speech enhancement network built upon the U-net architecture. Our approach incorporates a novel Multi-path Enhanced Taylor (MET) Transformer block, which integrates Deformable Embedding (DE) to enable flexible receptive fields for voiceprints. The MET Transformer is uniquely designed to fuse Channel and Spatial Attention (CSA) branches, facilitating channel information exchange and addressing spatial attention deficits within the Taylor-Transformer framework. Through extensive experiments conducted on the VoiceBank+DEMAND dataset, we demonstrate that MUSE achieves competitive performance while significantly reducing both training and deployment costs, boasting a mere 0.51M parameters.

MUSE was accepted by Interspeech 2024. [arxiv](https://arxiv.org/pdf/2406.04589) -->
## Pre-requisites
1. Clone this repository.
2. 运行以下命令会创建一个名为imse的conda环境，并且在imse环境中安装模型运行所需的python requirements。
```
conda env create -f environment.yml
```
4. Download and extract the [VoiceBank-DEMAND-16k](https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k). 使用 downsampling.py 去处理数据集，使数据集符合训练模型所需的格式，请认真读一下代码，了解运行 downsampling.py 的clean_train_path、noisy_train_path、clean_test_path、resample_path这四个变量，调整为你的项目目录后运行。
```
python downsampling.py
```
5. 运行 make_file_list.py 去获得training.txt和test.txt文件去进行模型训练
```
python make_file_list.py --train_clean_path "your train_clean_path" --train_noisy_path "your noisy_train_path" --test_clean_path "your clean_test_path" --test_noisy_path "your noisy_test_path"
```

## Training
For single GPU (Recommend), MUSE needs at least 8GB GPU memery.
```
python train.py --config config.json --input_training_file training.txt --input_validation_file test.txt
```


## Inference
1. 先使用 inference.py 去获得你的模型推理得到的降噪过后的 noisy_wav_dir
```
python inference.py --input_clean_wavs_dir "your clean_test_path" --input_noisy_wavs_dir "your noisy_test_path" --output_dir "your generated_files" --checkpoint_file "your g_best_pesq_3.xx"
```
2. 使用 create_test_list.py 去获得与 "your generated_files" 对应的test_final.txt文件在运行 cal_metrics.py 中使用
```
python create_test_list.py --test_dir_to_scan "your generated_files"
```
3. 使用 cal_metrics.py 去计算你的最优模型的 PESQ CSIG CBAK COVL STOI 指标
```
python cal_metrics.py --clean_wav_dir "your clean_test_path" --noisy_wav_dir "your generated_files" --input_test_file "your test_final.txt"
```
You can also use the pretrained best checkpoint file we provide in `paper_result/g_best_pesq_3.373`.<br>



## Acknowledgements
We referred to [MP-SENet](https://github.com/yxlu-0102/MP-SENet), [MB-TaylorFormer](https://github.com/FVL2020/ICCV-2023-MB-TaylorFormer)
