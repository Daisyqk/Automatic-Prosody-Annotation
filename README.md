# Automatic Prosody Annotation with Pre-Trained Text-Speech Model
This is the official PyTorch implementation of the following paper:

> **Automatic Prosody Annotation with Pre-Trained Text-Speech Model** \
> Ziqian Dai, Jianwei Yu, Yan Wang, Nuo Chen, Yanyao Bian, Guangzhi Li, Deng Cai, Dong Yu

> **Abstract**: *Prosodic boundary plays an important role in text-to-speech synthesis (TTS) in terms of naturalness and readability. However, the acquisition of prosodic boundary labels relies on manual annotation, which is costly and time-consuming. In this paper, we propose to automatically extract prosodic boundary labels from text-audio data via a neural text-speech model with pre-trained audio encoders. This model is pre-trained on text and speech data separately and jointly fine-tuned on TTS data in a triplet format: \{speech, text, prosody\}. The experimental results on both automatic evaluation and human evaluation demonstrate that: 1) the proposed text-speech prosody annotation framework significantly outperforms text-only baselines;  2) the quality of automatic prosodic boundary annotations is comparable to human annotations; 3) TTS systems trained with model-annotated boundaries are slightly better than systems that use manual ones.*
<!-- ![framework](framework.png,p_50) -->
<div align="center"><img src="https://github.com/Daisyqk/Automatic-Prosody-Annotation/blob/master/framework.png" width="600px"></div>

Visit our [demo page](https://daisyqk.github.io/Automatic-Prosody-Annotation_w/) for audio samples.


This implementation supports out prosody estimation model and the code for inference. Note that the model we provide here is "Conformer-Char" model in the paper.


## Getting Started

### Dependencies
The model is implemented with Python 3.6 and a couple of packages. In stall them use
```bash
pip install -r requirements.txt
```

### I. Data Preparation ### 
The audio data needs to be preprocessed before fed into the model. Firstly, install kaldi tookit. Then extract audio features with kaldi using the following command.
```bash
echo "input raw_audio.wav" > tmp.scp
compute-fbank-feats --num-mel-bins=80 scp:tmp.scp ark:fbk.ark
compute-kaldi-pitch-feats scp:tmp.scp ark:- | process-kaldi-pitch-feats ark:- ark:pitch.ark
paste-feats --length-tolerance=3 ark:fbk.ark ark:pitch.ark ark,scp:feature.ark,feature.scp
```
This command saves the extracted features in feature.ark, which can be read through feature.scp.

### II. Inference ### 
Download the project, put the features files mentioned above in folder “data”， then run the inference code by 
```bash
python code/main.py
```
The result will be stored in "prediction_save/test.txt". Note that the label 0-5 corresponds to CC, LW, PW, PPH, IPH in the paper respectively.
