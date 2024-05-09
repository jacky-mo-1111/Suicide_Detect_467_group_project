# Suicide_Detect_467_group_project
Suicide_Detect_467_group_project

Link to dataset: https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch 


The process for results for Naive Bayes are included in the link in the colab repository. Step by step instruction is included in the collab file. 

To train and eval fine-tuned roberta-base model, we need to run the following:

python run_roberta.py \
  --seed 1 \
  --model_name roberta-base \
  --dropout 0.1 \
  --chckpt_dir roberta_finetuned/ \
  --do_eval \

results will be saved in result.txt. 

To generate the predictions for gpt-3.5-turbo, we need to run the following:
python run_gpt.py \
    --shots_num 2 \

Prompts, demonstrations, prediciton and original label for each instance will be generated in shot_#_test_with_predictions. Ane the evaluation metrics can be computed based on whether the original label match the prediction. 
