import splitfolders
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, default="./chxnelset_example/bags/", help="dataset to split from")
parser.add_argument("-o", "--output", type=str, default="./chxnelset_bags_split", help="Destination directory")
args = vars(parser.parse_args())

# splits dataset into train and validation folders with predetermined ratio
# This script will split images under "chxnelset-example/watches" an copy into "chxnelset_watches_split"
splitfolders.ratio(args["input"], output=args["output"],
    seed=1221, ratio=(.8, .2), group_prefix=None, move=False)