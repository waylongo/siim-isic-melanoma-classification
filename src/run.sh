echo "<===== Program starts =====>"

echo "====== Traning CV... ======="
python train.py --image_size=512 --efficient_type=4 --batch_size=12 --random_seed=2012

echo "====== Inferencing... ======="
python test.py --image_size=512 --efficient_type=4 --batch_size=12 --random_seed=2012

echo "<===== Program ends =====>"

kaggle competitions submit -c siim-isic-melanoma-classification -f ../ensembles/preds/pred_imgsize_512_eff_b4.csv -m "512x512, eff-b4, ex2018"
echo "<===== Submission submitted to Kaggle =====>"