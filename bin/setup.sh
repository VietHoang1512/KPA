#!bin/bash

pip install -r requirements.txt

[[ -d kpm_6_folds ]] || ((gdown --id 1UawLUw-3dKYEpYaRhmo8l6eb8xlSksLB) && (unzip "kpm_6_folds.zip") && rm "kpm_6_folds.zip")

[[ -d IBM_Debater ]] || ((gdown --id 1ZbXhnhGOT8vmPJ6mVqI_fX4q5ztoVGR_) && (unzip "IBM_Debater.zip") && rm "IBM_Debater.zip")
