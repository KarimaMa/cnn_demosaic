import argparse
import os
import shutil


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Provide image directory')
    parser.add_argument('--input_dir', type=str, help='the path of the input image directory')
    parser.add_argument('--output_dir', type=str, help='the path of the output image directory')
    parser.add_argument('--n', type=int, help='number of images to transform')

    args = parser.parse_args()
    
    count = 0
    for filename in os.listdir(args.input_dir):
        count += 1
        shutil.move(os.path.join(args.input_dir, filename), os.path.join(args.output_dir, filename))                
        print(count)
        if count >= args.n:
            break      

