# Fashion-Hiernet
Fashion-Hiernet is a hierarchical image classification model for fashion commerce items based on EfficientNet-b4 and LCPN (Local Classifier per Parent Node) technique.

## venv setup

```bash
python3 -m venv venv
source ./venv/bin/activate
pip3 install -r requirements.txt
```

## Usage

### Dataset Preparation

For each parent node, run the following script to divide the dataset into train and validation set
```bash
python3 splitdata.py -i INPUT_DIR -o OUTPUT_DIR 
```
### Train Classifier
```bash
python3 main.py -tr OUTPUT_DIR/train -te OUTPUT_DIR/val
```
`bins` folder will include crucial pickle files specific to the parent node

`outputs` folder will contain the results after training is complete

### Predict
You need to manually move the generated `bin` folder into the outputs folder to use the following script.
```bash
python3 predict.py -p ../outputs
```

## Results
![](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/a1684197-640b-4e4d-9d15-29df19b6ff4b/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220621%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220621T072811Z&X-Amz-Expires=86400&X-Amz-Signature=9d2faeeae011d04a8731c74d8a699f01c1079838362a9aaafc521e573e0cef13&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject)

Avg. Acc: 80.18%

## License
[MIT](https://choosealicense.com/licenses/mit/)