#!/bin/bash

for i in 10
do
  ls /Users/kma/0$i >> 0$i"_image.txt"
  cat 0$i"_image.txt" | gshuf >> 0$i"_shuffled_image.txt"
  head -800 0$i"_shuffled_image.txt" >> 0$i"_train_image.txt"
  tail -200 0$i"_shuffled_image.txt" >> 0$i"_test_image.txt"
  mkdir "train_0"$i
  mkdir "test_0"$i
  python3 gen_bayer.py --input_dir="/Users/kma/0"$i --output_dir="train_0"$i --image_list="0"$i"_train_image.txt"
  python3 gen_bayer.py --input_dir="/Users/kma/0"$i --output_dir="test_0"$i --image_list="0"$i"_test_image.txt" 
  ls "train_0"$i >> "0"$i"_train_ids.txt"
  ls "test_0"$i >> "0"$i"_test_ids.txt"
  cat "0"$i"_train_ids.txt" > "_.txt"
  sed -e 's/^/\/Users\/kma\/cnn_demosaic\/train_0'$i'\//' $line < "_.txt" >> "train_ids.txt"
  cat "0"$i"_test_ids.txt" > "_.txt"
  sed -e 's/^/\/Users\/kma\/cnn_demosaic\/test_0'$i'\//' $line < "_.txt" >> "test_ids.txt"

done
