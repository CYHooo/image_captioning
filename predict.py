import torch
from PIL import Image
from transformers import AutoModelForCausalLM, CLIPImageProcessor, BertTokenizer
from transformers import AutoProcessor
import os
from glob import glob
import pandas as pd
from train import GitProcessor

def predict(premodel, processor, model_path, img_path, save_path, save_name=None):
    '''
        Args:
            premodel: pre-trained model
            processor: processor should same with triaining
            model_path: `str`, *.pth model file path
            img_path: `str`, predict images path
            save_path: `str`, file path for save result. default: `save_path + 'result.csv'`
            save_name: `str or None`, result csv file name, should end with `.csv`. if `None` default save as `result.csv`.
        Return:
            None
    '''

    model = premodel
    model.load_state_dict(torch.load(model_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # load predict images from path
    imgs = sorted(glob(os.path.join(img_path, '*.JPG')))
    result = []
    for i in imgs:
        if i.lower().endswith(('.jpg','.png','.jpeg')):
            image = Image.open(i).convert("RGB")  
            inputs = processor(images=image, return_tensors="pt")

            inputs = {k: v.to(device) for k, v in inputs.items()}
            pixel_values = inputs["pixel_values"]
            generated_ids = model.generate(pixel_values=pixel_values, max_length=200)
            generated_caption = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            print(f"\033[34mGenerated Caption img_{i}:\033[0m {generated_caption}")
            result.append({"filename":os.path.basename(i), "caption": generated_caption})
        else:
            print(f"{i} is not an image")

    df = pd.DataFrame(result, columns=["filename", "caption"])
    os.makedirs(save_path, exist_ok=True)

    if save_name is not None and save_name.lower().endswith('.csv'):
        print(f'\n\033[32m result saving to path \033[0m "{save_path}{save_name}"')
        df.to_csv(os.path.join(save_path, save_name), index=False)
    else:
        print(f'\n\033[32m result saving to path \033[0m "{save_path}result.csv"')
        df.to_csv(os.path.join(save_path,'result.csv'), index=False)



if __name__ == "__main__":
    premodel = AutoModelForCausalLM.from_pretrained("microsoft/git-base")

    ## GIT-BASE default processor
    # processor = AutoProcessor.from_pretrained("microsoft/git-base") 

    ## GIT && BERT processor
    image_processor = CLIPImageProcessor.from_pretrained('microsoft/git-base')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    processor = GitProcessor(image_processor, tokenizer)

    ## predict data path
    model_path = 'model/ep50_AdamW_bs8_attentaionmask_lr_5e5/best.pth'
    predict_imgs = 'data/predict/'
    save_path = 'data/result/'

    predict(premodel, processor, model_path, predict_imgs, save_path, save_name='GIT_ep50_bs8_attentaionmask_best.csv')

    