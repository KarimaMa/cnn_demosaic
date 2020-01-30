#!/bin/bash
for i in 0 1 2 3 4 5 6 7 8 9
do
  : '
  ls /Users/karima_ma/hdrvdp_data/00$i > 00$i"_image.txt"
  cat 00$i"_image.txt" | gshuf > 00$i"_shuffled_image.txt"
  head -8000 00$i"_shuffled_image.txt" > 00$i"_dense_train_image.txt"
  tail -2000 00$i"_shuffled_image.txt" > 00$i"_dense_test_image.txt"
  mkdir "train_00"$i
  mkdir "test_00"$i
  python3 gen_dense_bayer.py --input_dir="/Users/karima_ma/hdrvdp_data/00"$i --output_dir="train_00"$i --image_list="00"$i"_dense_train_image.txt"
  python3 gen_dense_bayer.py --input_dir="/Users/karima_ma/hdrvdp_data/00"$i --output_dir="test_00"$i --image_list="00"$i"_dense_test_image.txt" 
  ls "train_00"$i >> "00"$i"_dense_train_ids.txt"
  ls "test_00"$i >> "00"$i"_dense_test_ids.txt"
  '
  cat "00"$i"_dense_train_ids.txt" > "_.txt" 
  sed -e 's/^/\/Users\/karima_ma\/gitrepos\/cnn_demosaic\/train_00'$i'\//' $line < "_.txt" >> "dense_train_ids.txt" 
  cat "00"$i"_dense_test_ids.txt" > "_.txt"
  sed -e 's/^/\/Users\/karima_ma\/gitrepos\/cnn_demosaic\/test_00'$i'\//' $line < "_.txt" >> "dense_test_ids.txt"
done

for i in 10 11 12
do
  : '
  ls /Users/karima_ma/hdrvdp_data/0$i > 0$i"_image.txt"
  cat 0$i"_image.txt" | gshuf > 0$i"_shuffled_image.txt"
  head -8000 0$i"_shuffled_image.txt" > 0$i"_dense_train_image.txt"
  tail -2000 0$i"_shuffled_image.txt" > 0$i"_dense_test_image.txt"
  mkdir "train_0"$i
  mkdir "test_0"$i
  python3 gen_dense_bayer.py --input_dir="/Users/karima_ma/hdrvdp_data/0"$i --output_dir="train_0"$i --image_list="0"$i"_dense_train_image.txt"
  python3 gen_dense_bayer.py --input_dir="/Users/karima_ma/hdrvdp_data/0"$i --output_dir="test_0"$i --image_list="0"$i"_dense_test_image.txt" 
  ls "train_0"$i > "0"$i"_dense_train_ids.txt"
  ls "test_0"$i > "0"$i"_dense_test_ids.txt"
  '
  cat "0"$i"_dense_train_ids.txt" > "_.txt"
  sed -e 's/^/\/Users\/karima_ma\/gitrepos\/cnn_demosaic\/train_0'$i'\//' $line < "_.txt" >> "dense_train_ids.txt"
  cat "0"$i"_dense_test_ids.txt" > "_.txt"
  sed -e 's/^/\/Users\/karima_ma\/gitrepos\/cnn_demosaic\/test_0'$i'\//' $line < "_.txt" >> "dense_test_ids.txt"
done
