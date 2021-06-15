#!bin/bash

pip install -r requirements.txt

[[ -d kpm_6_folds ]] || ((gdown --id 1eDjsAyAgAqOKiYEOIRkMZ87B_J7xycTj) && (unzip "kpm_6_folds.zip") && rm "kpm_6_folds.zip")