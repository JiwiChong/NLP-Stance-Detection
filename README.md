# LLM-Stance-Detection

Stance Detection is a task in which textual processing and interpretability is made in order
to observe the viewpoint and stance that a person makes regarding a target. The target
can range from politicians, environmental campaigns, marketing campaigns, abortion, etc. 
In this repository, a research work of Machine Committee on Stance Detection is provided
in the hopes to promote further research work on Stance Detection. 

<div align="center">
	<img src="https://github.com/user-attachments/assets/3761da01-d322-46e6-93de-cca9fdfb40a3" width=80% height=80%>
</div><br />

Particularly, a Decorrelated Ensemble Model composed of multiple BERT models is developed. 
BERT models are the committee members, and they learn by supporting each other by each addressing 
different parts of the dataset. This is done so by having each model's learning patterns be
decorrelated from what others attenpt to detect, ensuring the Ensemble Model is learned to generalize better
across the dataset.

<div align="center">
<img src="https://github.com/user-attachments/assets/08785f73-c73f-4882-ae58-ad2f355ee9f6" width=85% height=90%>
</div><br />

The BERT Decorrelated Ensemble Model outperformed every original state-of-the-art Machine Learning and
Deep Learning algorithms for both Semeval-2016 and Will-they-won't-they datasets.

**_Semeval-2016_** <br />

<div align="center">
<img src="https://github.com/user-attachments/assets/b32be9de-c8dc-4c04-bba7-e073cb5d071b" width=80% height=90%>
</div><br />

**_Will-they-won't-they_** <br />

<div align="center">
<img src="https://github.com/user-attachments/assets/2516496f-bc0b-43c6-a535-5b1f725bdabb" width=80% height=90%> 
</div><br />

In addition, an explainable Stance Detection is executed in order to illustrate key words that
led to the Stance of the texts. 

### Commands
**_Train and Evaluation_** <br />
```python
python main.py --main_dir (Directory of repository) --num_experts (# of models in Ensemble) --task (Data used) --lambda_ (Lambda value in loss func) --emb_max_len (Max length of BERT embedding) --num_classes (# of output classes) --run_num (# of run)
```


