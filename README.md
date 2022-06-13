Implementation code for 

[Dynamical Hyperparameter Optimization via Deep Reinforcement Learning in Tracking.](https://www.researchgate.net/publication/337644592_Dynamical_Hyperparameter_Optimization_via_Deep_Reinforcement_Learning_in_Tracking) 
IEEE Trans. on Pattern Analysis and Machine Intelligence (TPAMI), 2019.

By Xingping Dong, Jianbing Shen, Wenguan Wang, Ling Shao, Haibin Ling, Fatih Porikli.

========================================================================

Any comments, please email: xingping.dong@gmail.com,
                            shenjianbingcg@gmail.com

This software was developed under Ubuntu 14.04 with python 2.7.

If you use this software for academic research, please consider to cite the following papers:

[1] Xingping Dong, Jianbing Shen, Wenguan Wang, Ling Shao, Haibin Ling, Fatih Porikli.
Dynamical Hyperparameter Optimization via Deep Reinforcement Learning in Tracking. IEEE Trans. on Pattern Analysis and Machine Intelligence (TPAMI), 2019, in press, DOI: 10.1109/TPAMI.2019.2956703. 

[2] Xingping Dong, Jianbing Shen, Wenguan Wang, Yu Liu, Ling Shao, Fatih Porikli. 
Hyperparameter optimization for tracking with continuous deep q-learning. In IEEE CVPR, pp. 518-527. 2018.

[3] Xingping Dong, Jianbing Shen, Dongming Wu, Kan Guo, Xiaogang Jin, Fatih Porikli. 
Quadruplet network with one-shot learning for fast visual object tracking. IEEE Trans. on Image Processing (TIP), 2019 Feb 11;28(7):3516-27.

[4] Xingping Dong, Jianbing Shen. 
Triplet loss in siamese network for object tracking. In ECCV, pp. 459-474. 2018.

**BIB**
```bibtex
@article{dong2019dynamical,
  title={Dynamical Hyperparameter Optimization via Deep Reinforcement Learning in Tracking},
  author={Dong, Xingping and Shen, Jianbing and Wang, Wenguan and Shao, Ling and Ling, Haibin and Porikli, Fatih},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2019},
  publisher={IEEE}
}
@inproceedings{dong2018hyperparameter,
  title={Hyperparameter optimization for tracking with continuous deep q-learning},
  author={Dong, Xingping and Shen, Jianbing and Wang, Wenguan and Liu, Yu and Shao, Ling and Porikli, Fatih},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={518--527},
  year={2018}
}
@article{dong2019quadruplet,
  title={Quadruplet network with one-shot learning for fast visual object tracking},
  author={Dong, Xingping and Shen, Jianbing and Wu, Dongming and Guo, Kan and Jin, Xiaogang and Porikli, Fatih},
  journal={IEEE Transactions on Image Processing},
  volume={28},
  number={7},
  pages={3516--3527},
  year={2019},
  publisher={IEEE}
}
@inproceedings{dong2018triplet,
  title={Triplet loss in siamese network for object tracking},
  author={Dong, Xingping and Shen, Jianbing},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={459--474},
  year={2018}
}
```

[**Prerequisites**]

keras = 2.1.6 (pip install keras==2.1.6)

tensorflow >=1.6.0 (https://www.tensorflow.org/install/?hl=zh-cn, pip install tensorflow-gpu==1.6.0)

Keras rl (https://github.com/keras-rl/keras-rl,pip install keras-rl)

gym >=0.9.2 (https://github.com/openai/gym, pip install gym)

PIL (pip install pillow)

matplotlib (pip install matplotlib)


[**Install**]

python ./setup.py install (install the env for tracker)

[**Training**] After you finish all Prerequisites, you can train our network step by step.

  1. Perpare dataset:(similar instructions in SiamFC:'Fully-Convolutional Siamese Networks for Object Tracking' )
	1. Signup [here](http://image-net.org/challenges/LSVRC/2015/signup) to obtain the link to download the data of the 2015 challenge.
	2. Download and unzip the full original ImageNet Video dataset (the 86 GB archive).
	3. Move `ILSVRC15/Data/VID/validation` to `ILSVRC15/Data/VID/train/` so that inside `train/` there are 5 folders with the same structure. You had better to rename these folders and use very short names ( such as a, b, c, d, e) in order to use the prepared metadata.
	4. If you did not generate your own, use 10 small metadatas (such as './ILSVRC2015_small_imdbs/imdb_video_01.mat') splited from [imdb_video.mat](http://bit.ly/imdb_video). You can also generate your own `imdb_video.mat` following these [step-by-step instructions](https://github.com/bertinetto/siamese-fc/tree/master/ILSVRC15-curation).
  
  2. Edit the file 'data_paths.json' to setup the paths in your machine:
	"imdb_path":"/path/to/ILSVRC2015_small_imdbs/imdb_video_01.mat",
	"imdb_name":"imdb_video_small",
	"imdb_video_path":"/path/to//ILSVRC2015/Data/VID/train/",
	"imdb_score_map_path":"/path/to/ILSVRC2015_score_map/Data/VID/train/", (saving the score maps)
	"imdbs_path":"/path/to/ILSVRC2015_small_imdbs/" 
  3. Edit the file './gym_hyper/envs/siam_src/parameters/environment.json' to setup the paths in your machine:
  4. Run the script for training:
	cd /path/to/code_folder/
	python ./run_training.py
  5. After training, you can find the trained mu model in './models/*mu_model.h5'. Then this model can be used for tracking in next step.

[**Tracking**]

  1. You can directly run './run_tracker_hp_evaluation_test.py' as a demo to test our algrithom with the pre-trained model ('./models/mu_model.h5'). You can also try your trained model by modifying the model path 'mu_path' in this file.
  2. You can also evaluate the OTB-2013 dataset by modifying the database path 'dataset_folder' in the file './run_tracker_hp_evaluation_otb.py' and then run this file for evaluation. Similarly, you can also modify the model path 'mu_path' to test your trained model.

[**Results**]

1. You can find the results of our paper [1] in 'results in paper.zip'.
