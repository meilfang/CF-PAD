# CF-PAD


---
## Note
This is the official repository of the paper: Face Presentation Attack Detection by Eacavating Causal Clues and Adapting Embedding Statistics, accepted at WACV 2024.. The paper can be found in [here](https://arxiv.org/abs/2308.14551).

## Pipeline Overview
![overview](images/workflow)

## Data preparation
Since the data in most used PAD datasets in our work are videos, we sample 25 frames in the average time interval of each video. Then, the face is extracted by using MTCNN. CSV files are generated for further training and evaluation. The format of the dataset CSV file is:
```
image_path,label
/image_dir/image_file_1.png, bonafide
/image_dir/image_file_2.png, bonafide
/image_dir/image_file_3.png, attack
/image_dir/image_file_4.png, attack
```


## Training
Example of training:
    ```
    python train.py \
      --training_csv train.csv \
      --test_csv test.csv
      --prefix 'custom_note' \
    ```
Check --ops for different counterfactual interventions.

## Testing
Example of testing:
```
python test.py \
  --test_csv 'test_data.csv' \
  --model_path 'model.pth'
```
where test_data.csv contains image path and the corresponding label (bonafide or attack).


## Results
PAD performance vs. Computational complexity. More details can be found in paper.
![cross_db](images/computation_complexity.pdf)

## Models
Four models pre-trained based on four cross-dataset experimental settings can be download via [google driver](https://drive.google.com/drive/folders/1E_u3nW3vux9f0gi2lNf5Kb5MVy2J1BWy?usp=sharing).
More information and small test can be found in test.py. Please make sure give the correct model path.

if you use CF-PAD in this repository, please cite the following paper:
```
@inproceedings{cf_pad,
  author       = {Meiling Fang and
                  Naser Damer},
  title        = {Face Presentation Attack Detection by Eacavating Causal Clues and Adapting Embedding Statistics},
  booktitle    = {{WACV}},
  publisher    = {{IEEE}},
  year         = {2024}
}
```


## License
This project is licensed under the terms of the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license. Copyright (c) 2020 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt.
