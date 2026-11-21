from fastapi import FastAPI, Request
from pydantic import BaseModel 
from transformers import T5ForConditionalGeneration, T5Tokenizer 
import torch 
import re
from fastapi.templating import Jinja2Templates 
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles 


# initialize our fastapi app 
app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

# load the model & tokenizer 
model = T5ForConditionalGeneration.from_pretrained("./my_summarizer_model")
tokenizer = T5Tokenizer.from_pretrained("./my_summarizer_model")

# device 
if torch.backends.mps.is_available() :
    device = torch.device("mps")
elif torch.cuda.is_available() :
    device = torch.device("cuda")
else :
    device = torch.device("cpu")        
   
model.to(device)   
   
# templating 
templates = Jinja2Templates(directory=".") # by dot(.) we mean that we tell the fastapi that the index.html is available in this very folder itself

# Input schema for dialogue => string 
class DialogueInput(BaseModel) :
    dialogue : str    
    
def clean_data(text) :
    text = re.sub(r"\r\n", " ", text) # lines 
    text = re.sub(r"\s+", " ", text) # spaces
    text = re.sub(r"<.*?>", " ", text) # html tags
    text = text.strip().lower()
    return text   


# Test the core logic for Summarization 
def summarize_dialogue(dialogue : str) -> str :
    # clean 
    dialogue = clean_data(dialogue) 
    
    # tokenize 
    inputs = tokenizer(
        dialogue,
        padding = "max_length",
        max_length = 512,
        truncation = True,
        return_tensors = "pt"
    )
    # generate the summary => token ids 
    model.to(device) # to ensure that the model & the data are on the same device
    targets = model.generate(
        input_ids = inputs["input_ids"],
        attention_mask = inputs["attention_mask"],
        max_length = 150,
        num_beams = 4,
        early_stopping = True
    )
    
    # convert token ids to summary ie decoding our output
    summary = tokenizer.decode(targets[0], skip_special_tokens = True) # special tokens such as EOS, seoarators, tab spaces etc
    return summary
 
# API Endpoints 
@app.post("/summarize/")
async def summarize(dialogue_input : DialogueInput) :
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary" : summary}

@app.get("/", response_class = HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request, "summary": ""})