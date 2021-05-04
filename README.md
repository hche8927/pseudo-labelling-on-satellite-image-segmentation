# Pseudo Labelling on Satellite Image Segmentation


## Introduction
This project tests the effect of pseudo labelling on semantic segmentation deep learning models using satellite images. Two pseudo labelling methods were tested, the original method (Lee, 2013) and the teacher-student paradigm approach (Yalniz et al., 2019). The psuedo-label data balenced version of these two methods were also tested. The dataset comes from 2016 UK Defence Science and Technology Laboratory (DSTL) Kaggle challenge (DSTL, 2016). 


## Prerequisite
1. Craete a new python venv.
2. Install packages from requirement.txt
3. Go to DSTL Kaggle challenge page and download the data.  
*All experiments are done in jupyter notebooks.*


## Run code for yourself
5. Set data root directory in each jupyter notebook to the specified location.
6. Open file `src/data_preprocessing.ipynb` and execute all code blocks to prepare/generate data for training. (Some code may need to be un-commented.)
7. Run desired notebooks. (See file structure below.)


## File Structure Explanation
```
.
├─lib <--- Models implemented by others
│
├─src 
│  │  data_preprocessing.ipynb       <--- Prepare data for training
│  │  project_utils.py               
│  │  pseudo_label_balanced.ipynb    <--- Balanced teacher-student paradigm
│  │  pseudo_label_basic.ipynb       <--- Base pseudo labelling
│  │  pseudo_label_ts.ipynb          <--- Base teacher-student paradigm
│  │  pseudo_label_ts_balanced.ipynb <--- Balanced teacher-student paradigm
│  │  SatelliteImageData.py          
│  │  segmentation_models.ipynb      <--- Base models
│  │
│  └─saved_weights
│
├─results_visualization
│  │  visualise_results.ipynb
│  └─experiment_results
```

## Experement Results Overview
![Alt text](results_visualization/dl_vs_pl.png?raw=true)


## References
- Dstl Satellite Imagery Feature Detection | Kaggle. (2016). Kaggle.com. https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/data
- Lee, Dong-Hyun. (2013). Pseudo-Label : The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks. ICML 2013 Workshop : Challenges in Representation Learning (WREPL). 
- Yalniz, I. Z., Jégou, H., Chen, K., Paluri, M., & Mahajan, D. (2019). Billion-scale semi-supervised learning for image classification. arXiv preprint arXiv:1905.00546.

