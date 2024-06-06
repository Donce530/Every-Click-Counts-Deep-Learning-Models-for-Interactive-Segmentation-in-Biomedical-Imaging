# Every Click Counts: Deep Learning Models for Interactive Segmentation in Biomedical Imaging

This repostory contains the python code and shell scripts used for the Interactive segmentation methods used in the Master's thesis of Donatas Vaiciukevičius. The code is based on the repositories of [RITM](https://github.com/SamsungLabs/ritm_interactive_segmentation) and [FocalClick](https://github.com/XavierCHEN34/ClickSEG).

The bash scripts for training and evalution of RITM and FocalClick models can be found in the *trainval_scripts* directory.

Main logic for training and inference of the model is implemented in pytorch and is located in *isegm* directory.

Directory *notebooks* contains various Jupyter Notebooks used for experimentation with design of the models, qualitative evaluation and debugging.

Required python packages can be installed by running `pip install -r requirements.txt`
