# NER_model_training
NER model training instructions

Please follow below instructions to Train NER model on training/testing files:

### SCRIPT MODE
	
1. Make sure all these files are present in your data directory (train.txt, train_box.txt, train_image.txt, test.txt, test_box.txt, test_image.txt, labels.txt) /

2. Run following command to train NER model for python script based:
!python ./train.py --epochs=5 --data_dir='./data' --output_dir='./out' --train_batch_size=4 --eval_batch_size=4 --checkpoint_model_path='./output/checkpoints/' /
Parameters: /
* Epochs - Number of epochs to train model /
* data_dir - data directory where training and test files are present(train.txt, train_box.txt, train_image.txt, test.txt, test_box.txt, test_image.txt, labels.txt) /
* output_dir - output directory where all model output files are stored like model checkpoint, model metrics on test data, best model checkpoint used for inference/deployment. /
* train_batch_size - train batch size (Adjust as per GPU memory) /
* eval_batch_size - test batch size (Adjust as per GPU memory) /
* Checkpoint_model_path - pass last model checkpoint files to start model training from this checkpoint /

Note:
 * Please refer to this link for demonstration - https://drive.google.com/drive/u/2/folders/1tyWBXTPJjSQGGN9O5Cm-5OJow4Nzorp7 /
 * Open and run train_layoutlm_script.ipynb notebook cells. /



### NOTEBOOK TRAINING

1. Make sure all these files are present in your data directory (train.txt, train_box.txt, train_image.txt, test.txt, test_box.txt, test_image.txt, labels.txt)/
2. Open and run LayoutLMv1_training_script.ipynb notebook cells sequentially./
3. Change following directories path in notebook cells:/
* Labels.txt file directory- /
  labels = get_labels("/content/drive/My Drive/aws_annotation/data/labels.txt") /
  please follow this link to refer - https://colab.research.google.com/drive/1sn0xCIbuOEz6GgWo3lnhaMY-VNWHW-o8?authuser=2#scrollTo=PIjINVaNRl8J line=14&uniqifier=1 /
* Data directory - /
  args = {'local_rank': -1, /
  'overwrite_cache': True, /
  'data_dir': '/content/drive/MyDrive/aws_annotation/data', /
  'model_name_or_path':'microsoft/layoutlm-base-uncased', /
  'max_seq_length': 512, /
  'model_type': 'layoutlm',} /
  please follow this link to refer - https://colab.research.google.com/drive/1sn0xCIbuOEz6GgWo3lnhaMY-VNWHW-o8?authuser=2#scrollTo=9cBBA_ws0rTZ&line=5&uniqifier=1 /

* Model checkpoints and output directory -
    best_model = f'models/checkpointLM1_epoch{epoch}.pt' /
    for ckpt in os.listdir('models'): /
    if 'checkpointLM1_epoch' in ckpt: /
    os.remove(f'models/{ckpt}') /
    
    if not os.path.exists(f'results/v1'): /
    os.mkdir(f'results/v1') /
    df.to_csv(f'results/v1/log_v1_FUNSD_{test_mode}_run{run}.csv', index = False) /
    please follow this link to refer - https://colab.research.google.com/drive/1sn0xCIbuOEz6GgWo3lnhaMY-VNWHW-o8?authuser=2#scrollTo=JXsd7u37jPud&line=166&uniqifier=1  /

Note:  
    * Please refer to this link for demonstration- /
    * https://drive.google.com/drive/u/2/folders/1ykGJ3fD29gJYMgkZIsBYMp4shtk28YKc /
