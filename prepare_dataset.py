import argparse
import os
import shutil
import splitfolders

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", type=str, default="./chanelset", help="Dataset Relative Path")
args = vars(parser.parse_args())

DATAPATH = args["path"]

def create_parentset(oPath):
    nPath = oPath + "_parent"
    folder = [x for x in os.listdir(oPath) if os.path.isdir(os.path.join(oPath,x))]

    for cat in folder:
        directory = f"{cat}"

        xPath = os.path.join(nPath, directory)

        try:
            os.makedirs(xPath, exist_ok = True)
            print("Directory '%s' created successfully" % xPath)
        except OSError:
            print("Directory '%s' can not be created" % directory)

    for cat in folder:
        y = os.listdir(oPath +"/"+ cat)

        for file in y:
            z = os.listdir(oPath +"/"+ cat+"/"+ file)
            count = 0
            for fil in z:
                count += 1
                filenum = format(count, "04")
                original = f"{oPath}/{cat}/{file}/img_{filenum}.jpg"
                target = f"{nPath}/{cat}/{file}_{filenum}.jpg"
                shutil.copyfile(original, target)

def split_train_val(path, output, ratio=(.8, .2), seed=1337):
    splitfolders.ratio(path, output=output,
    seed=seed, ratio=ratio, group_prefix=None, move=False)


if __name__ == "__main__":
    # generates ./parent
    create_parentset(oPath=DATAPATH)
    split_train_val(path=DATAPATH+"_parent", output="./split_dataset/parent")
    shutil.rmtree(DATAPATH+"_parent")

    # generates ./dataset_{childclass}_split recursively
    folder = [x for x in os.listdir(DATAPATH) if os.path.isdir(os.path.join(DATAPATH,x))]
    for childclass in folder:
        split_train_val(path=f"{DATAPATH}/{childclass}", output=f"./split_dataset/{childclass}")