# GSum 修改

此 code 是由 `GSum: A General Framework for Guided Neural Abstractive Summarization` 的 [code](https://github.com/neulab/guided_summarization) 啟發的。

## 製作此 code 的原因

由於原論文的提供的模型架構與訓練腳本是使用 Fairseq 的等 api 進行撰寫並架設。Facebook 所提出的預訓練語言模型 BART 在英文的生成式摘要任務達到了 State-of-the-art，但研究過程想藉由復旦大學提供的中文版本之 BART [fnlp/bart-large-chinese](https://huggingface.co/fnlp/bart-large-chinese) 進行實驗，由於 checkpoint 提供在 HuggingFace 上，無法直接使用原論文提供的腳本進行實驗，因此藉由論文的架構與 GSum 提供的 code 中的 bart 版本進行參考，撰寫以下模型架構。

## 實驗環境使用

- PyTorch == 1.8.1
- Cuda == 10.1
- Transformers == 4.27.0

## 資料輸入格式

可使用 `.csv` 進行輸入，將資料分出 文本輸入(text_column)、引導輸入(guided_column)、摘要輸出(summary_column)。

資料切分為 train\val\test

## 訓練步驟

```bash
python model_preprocess.py
```

```bash
CUDA_VISIBLE_DEVICES="0" python train.py \
        --model_name_or_path /workplace/yhcheng/gsum_like/chinese_gsum_bart \
        --tokenizer_name fnlp/bart-large-chinese \
        --text_column Input \
        --guided_column Guidance \
        --summary_column Summary \
        --train_file data/demo.csv \
        --validation_file data/demo_val.csv \
        --test_file data/demo_test.csv \
        --max_source_length 512 \
        --max_target_length 256 \
        --pad_to_max_length true \
        --num_beams 4 \
        --output_dir output_models \
        --overwrite_output_dir \
        --do_train \
        --do_eval \
        --evaluation_strategy epoch \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --learning_rate 2e-5 \
        --weight_decay 0.001 \
        --num_train_epochs 20 \
        --warmup_ratio 0.1 \
        --warmup_steps 100 \
        --logging_dir logs \
        --logging_strategy epoch \
        --save_strategy epoch \
        --save_total_limit 5 \
        --fp16 \
        --label_smoothing_factor 0 \
        --predict_with_generate true \
        --seed 42
```
