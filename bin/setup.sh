#!bin/bash

pip install -r requirements.txt

[[ -d kpm_k_folds ]] || ((gdown --id 14Bf8zs4LMzAOGSjyXPro4py-qQdQYjI0) && (unzip "kpm_k_folds.zip") && rm "kpm_k_folds.zip")

[[ -d IBM_Debater ]] || ((gdown --id 1ZbXhnhGOT8vmPJ6mVqI_fX4q5ztoVGR_) && (unzip "IBM_Debater.zip") && rm "IBM_Debater.zip")
