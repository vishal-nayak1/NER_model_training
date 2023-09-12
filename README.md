# NER_model_training
NER model training instructions

Please follow below instructions to Train NER model on training/testing files:

### SCRIPT MODE
	
1. Make sure all these files are present in your data directory _(train.txt, train_box.txt, train_image.txt, test.txt, test_box.txt, test_image.txt, labels.txt)_ <br>

2. Run following command to train NER model for python script based: <br>
_!python ./train.py --epochs=5 --data_dir='./data' --output_dir='./out' --train_batch_size=4 --eval_batch_size=4 --checkpoint_model_path='./output/checkpoints/'_ <br>
Parameters: <br>
* __epochs__ - Number of epochs to train model <br>
* __data_dir__ - data directory where training and test files are present _(train.txt, train_box.txt, train_image.txt, test.txt, test_box.txt, test_image.txt, labels.txt)_ <br>
* __output_dir__ - output directory where all model output files are stored like model checkpoint, model metrics on test data, best model checkpoint used for inference/deployment. <br>
* __train_batch_size__ - train batch size (Adjust as per GPU memory) <br>
* __eval_batch_size__ - test batch size (Adjust as per GPU memory) <br>
* __checkpoint_model_path__ - pass last model checkpoint files to start model training from this checkpoint <br>

#### Note: <br>
 * Please refer to this link for demonstration - https://drive.google.com/drive/u/2/folders/1tyWBXTPJjSQGGN9O5Cm-5OJow4Nzorp7 <br>
 * Open and run _train_layoutlm_script.ipynb_ notebook cells. <br>



### NOTEBOOK TRAINING

1. Make sure all these files are present in your data directory (train.txt, train_box.txt, train_image.txt, test.txt, test_box.txt, test_image.txt, labels.txt)<br>
2. Open and run LayoutLMv1_training_script.ipynb notebook cells sequentially. <br>
3. Change following directories path in notebook cells:<br>
* __Labels.txt file directory__ - <br>
  * labels = get_labels("/content/drive/My Drive/aws_annotation/data/labels.txt") <br>
  
  * please follow this link - https://colab.research.google.com/drive/1sn0xCIbuOEz6GgWo3lnhaMY-VNWHW-o8?authuser=2#scrollTo=PIjINVaNRl8Jline=14&uniqifier=1 <br>
  
* __Data directory__ - <br>
  * args = {'local_rank': -1, <br>
  'overwrite_cache': True, <br>
  'data_dir': '/content/drive/MyDrive/aws_annotation/data', <br>
  'model_name_or_path':'microsoft/layoutlm-base-uncased', <br>
  'max_seq_length': 512, <br>
  'model_type': 'layoutlm',} <br>
  
  * please follow this link - https://colab.research.google.com/drive/1sn0xCIbuOEz6GgWo3lnhaMY-VNWHW-o8?authuser=2#scrollTo=9cBBA_ws0rTZ&line=5&uniqifier=1 <br>

* __Model checkpoints and output directory__ - <br>
    * best_model = f'models/checkpointLM1_epoch{epoch}.pt' <br>
    for ckpt in os.listdir('models'): <br>
    if 'checkpointLM1_epoch' in ckpt: <br>
    	os.remove(f'models/{ckpt}') <br>
    if not os.path.exists(f'results/v1'): <br>
        os.mkdir(f'results/v1') <br>
        df.to_csv(f'results/v1/log_v1_FUNSD_{test_mode}_run{run}.csv', index = False) <br>
    
    * please follow this link - https://colab.research.google.com/drive/1sn0xCIbuOEz6GgWo3lnhaMY-VNWHW-o8?authuser=2#scrollTo=JXsd7u37jPud&line=166&uniqifier=1  <br>
    
#### Note:  <br>
  * Please refer to this link for demonstration - https://drive.google.com/drive/u/2/folders/1ykGJ3fD29gJYMgkZIsBYMp4shtk28YKc <br>
