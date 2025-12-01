# IMSE: Efficient U-Net-based Speech Enhancement using Inception Depthwise Convolution and Amplitude-Aware Linear Attention
### Xinxin Tang Bin Qin Yufang Li


**Abstract:** 
Achieving a balance between lightweight design and high performance remains a significant challenge for speech enhancement (SE) tasks on resource-constrained devices. Existing state-of-the-art methods, such as MUSE, have established a strong baseline with only 0.51M parameters by introducing a Multi-path Enhanced Taylor (MET) transformer and Deformable Embedding (DE). However, MUSE suffers from structural redundancy due to its approximate-then-compensate attention mechanism and computationally expensive deformable offsets. We propose IMSE to resolve these bottlenecks via: 1) Amplitude-Aware Linear Attention (MALA), which fundamentally rectifies magnitude loss in linear attention, eliminating the need for auxiliary compensation branches; and 2) Inception Depthwise Convolution (IDConv), which efficiently captures the anisotropic features of spectrograms (e.g., harmonic strips) using decomposed static kernels instead of heavy dynamic deformations. Extensive experiments on the VoiceBank+DEMAND dataset demonstrate that IMSE significantly reduces the parameter count by 16.8% (from 0.513M to 0.427M) while achieving superior performance compared to the MUSE baseline. Specifically, it achieves a PESQ score of 3.399, setting a new state-of-the-art benchmark for ultra-lightweight speech enhancement models.

<!--   MUSE was accepted by Interspeech 2024. [arxiv](https://arxiv.org/pdf/2406.04589)       -->
## Pre-requisites
1. Clone this repository.
2. Run the following command to create a conda environment named imse and install the required Python dependencies in the imse environment.
```
conda env create -f environment.yml
```
3. activate conda env

```
conda activate imse
```

4. Download and extract the [VoiceBank-DEMAND-16k](https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k). Use downsampling.py to process the dataset to format it for model training. Please read the code carefully to understand the four variables required to run downsampling.py: clean_train_path, noisy_train_path, clean_test_path, and resample_path. Adjust them to your project directory and then run the script.
```
python downsampling.py
```
5.Run make_file_list.py to obtain the training.txt and test.txt files for model training.
```
python make_file_list.py --train_clean_path "your train_clean_path" --train_noisy_path "your noisy_train_path" --test_clean_path "your clean_test_path" --test_noisy_path "your noisy_test_path"
```

## Training
For single GPU (Recommend), MUSE needs at least 8GB GPU memery.
```
CUDA_VISIBLE_DEVICES=0 python train.py --config config.json --input_training_file training.txt --input_validation_file test.txt
```


## Inference
1. First, use inference.py to generate the denoised audio files from your model.
```
python inference.py --input_clean_wavs_dir "your clean_test_path" --input_noisy_wavs_dir "your noisy_test_path" --output_dir "your generated_files" --checkpoint_file "your g_best_pesq_3.xx"
```
2. Use create_test_list.py to generate the test_final.txt file corresponding to "your generated_files", which will be used when running cal_metrics.py.
```
python create_test_list.py --test_dir_to_scan "your generated_files"
```
3. Use cal_metrics.py to calculate the PESQ, CSIG, CBAK, COVL, and STOI metrics for your best model.
```
cd cal_metrics
```
```
python cal_metrics.py --clean_wav_dir "your clean_test_path" --noisy_wav_dir "your generated_files" --input_test_file "your test_final.txt"
```
You can also use the pretrained best checkpoint file we provide in `paper_result/g_best_pesq_3.399`.<br>



## Acknowledgements
We referred to [MP-SENet](https://github.com/yxlu-0102/MP-SENet), [MB-TaylorFormer](https://github.com/FVL2020/ICCV-2023-MB-TaylorFormer), [MUSE](https://github.com/huaidanquede/MUSE-Speech-Enhancement)
