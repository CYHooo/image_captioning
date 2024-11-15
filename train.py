import json
import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoModelForCausalLM, CLIPImageProcessor, BertTokenizer
import os 
from PIL import Image
from torch.utils.data import DataLoader
import torch 

def preprocess(csv_path, json_path):
    '''
        preprocess for make training .jsonl file    

        Args:
            csv_path: dataset label file for *.csv, *.text, *.excel
            json_path: training json file path. save as `json_path + 'data.jsonl'`

        Return:
            None
    '''

    caption = []
    data = pd.read_csv(csv_path)
    filename = data['filename']
    text = data['content']
    for f, t in zip(filename, text):
        if type(f) and type(t) is str:
            caption.append({"file_name":f,"text":t.lower()},)

    with open(json_path + "data.jsonl", 'w') as f:
        for item in caption:
            f.write(json.dumps(item) + "\n")


class GitDataset(Dataset):
    def __init__(self, dataset, processor, img_path):
        self.dataset = dataset
        self.processor = processor
        self.img_path = img_path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = Image.open(os.path.join(self.img_path ,item['file_name'])).convert("RGB")
        encoding = self.processor(images=image, text=item["text"], padding="max_length", return_tensors="pt")

        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        encoding["attention_mask"] = encoding["input_ids"].ne(0).long()

        return encoding

class GitProcessor:
    '''
        Processor with BERT-Tokenizer
    '''
    def __init__(self, image_processor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(self, images=None, text=None, padding="max_length", return_tensors="pt"):
        
        inputs = {}
        if images is not None:
            inputs["pixel_values"] = self.image_processor(images=images, return_tensors=return_tensors)["pixel_values"]

        if text is not None:
            text_inputs = self.tokenizer(text, padding=padding, return_tensors=return_tensors, truncation=True)
            inputs.update(text_inputs)

        return inputs


    
def train(dataset, model, epoch, batch_size, lr, save_path):
    '''
        Args:
            dataset: dataset
            model: pretrain model
            epoch: `int`, training epoch
            batch_size: `int`, batch size
            lr: `float`, learing rate
            save_path: `str`,save training model path

        Return:
            None
    '''
    os.makedirs(save_path, exist_ok=True)

    train_dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    batch = next(iter(train_dataloader))

    outputs = model(input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["input_ids"])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTraining on device: {device}\n")

    model.to(device)
    model.train()

    best_loss = float("inf") ## init best loss on inf
    for epoch in range(epoch):
        print("Epoch:", epoch)
        for idx, batch in enumerate(train_dataloader):
            input_ids = batch.pop("input_ids").to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch.pop("pixel_values").to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            labels=input_ids)
            
            loss = outputs.loss
            print("Loss:", loss.item())
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), f"{os.path.join(save_path, 'best.pth')}")
        

    torch.save(model.state_dict(), f"{os.path.join(save_path, 'last.pth')}")

if __name__ == "__main__":
    imgfolder = 'data/imgs/'
    labelfolder = 'data/lable/lable.csv'

    ## preprocess for generate label jsonl file
    preprocess(labelfolder, imgfolder) 

    dataset = load_dataset('json', data_files='data/imgs/data.jsonl', split="train")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")
    
    ## GIT default processor
    processor = AutoProcessor.from_pretrained("microsoft/git-base")

    ## GIT && BERT
    # image_processor = CLIPImageProcessor.from_pretrained('microsoft/git-base')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # processor = GitProcessor(image_processor, tokenizer)

    train_dataset = GitDataset(dataset, processor, imgfolder)

    ## parameter
    epoch = 50
    batch_size = 8
    lr = 5e-5
    save_path = 'model/ep50_AdamW_bs8_attentaionmask_lr_5e5/'
    train(dataset=train_dataset, model=model, epoch=epoch, batch_size=batch_size, lr=lr, save_path=save_path)
