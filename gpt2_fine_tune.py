! pip install transformers

### If you run for 10 epochs (training takes longer) you use all the checkpoints, otherwise you will have to use till checkpoint < step
## this is buiding model from scratch

from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
import torch

config = GPT2Config()

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', max_len=512)
tokenizer.add_tokens('<pad>')
tokenizer.pad_token = '<pad>'

model = GPT2LMHeadModel(config=config)

from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="../input/romcom/all.txt",
    block_size=128,
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)
model.resize_token_embeddings(len(tokenizer))


from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir="./GPT2new-2",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_gpu_train_batch_size=32,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    
)

trainer.train()

model.save_pretrained('./GPT2new-2')

"""# Fine-tune GPT-2 Model"""

from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
import torch

config = GPT2Config()

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', max_len = 512)
# tokenizer.add_tokens('<pad>')
tokenizer.pad_token = '<pad>'

model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)

from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="../input/romcom/all.txt",
    block_size=128,
)

tokenizer.add_tokens('<pad>')
tokenizer.pad_token = '<pad>'

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

model.resize_token_embeddings(len(tokenizer))

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./GPT2final",
    overwrite_output_dir=True,
    num_train_epochs=4,
    per_gpu_train_batch_size=32,
    save_steps=5_000,
    save_total_limit=25,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    
)

trainer.train()

trainer.save_model("./GPT2final")

### For using Checkpoints
device = 'cuda'

import numpy as np
checkpoints = ['checkpoint-5000', 'checkpoint-10000', 'checkpoint-15000', 'checkpoint-20000', 'checkpoint-25000', 'checkpoint-30000', 'checkpoint-35000', 'checkpoint-40000', 'checkpoint-45000', 
 'checkpoint-50000', 'checkpoint-55000', 'checkpoint-60000', 'checkpoint-65000', 'checkpoint-70000', 'checkpoint-75000', 'checkpoint-80000']
checkdict = {}

for i in checkpoints:
  checkdict[i] = GPT2LMHeadModel.from_pretrained('./GPT2final/' + i, config=config)
  checkdict[i] = checkdict[i].to(device)

checkdict['gpt'] = model
checkdict['gpt'] = checkdict['gpt'].to(device)

checkdict['final'] = GPT2LMHeadModel.from_pretrained('./GPT2final/', config=config)
checkdict['final'] = checkdict['final'].to(device)

checkdict['new'] = GPT2LMHeadModel.from_pretrained('./GPT2new-2/', config=config)
checkdict['new'] = checkdict['new'].to(device)


love_input = torch.tensor(tokenizer.encode("love", add_special_tokens=True)).unsqueeze(0)
love_input = love_input.to(device)
m = torch.nn.Softmax(dim=2)

plove = {}
for i in checkpoints:
  plove[i] = torch.distributions.categorical.Categorical(m(checkdict[i](love_input)[0]))

for i in ['gpt', 'final', 'new']:
  plove[i] = torch.distributions.categorical.Categorical(m(checkdict[i](love_input)[0]))

KLlovenew = torch.distributions.kl.kl_divergence(plove['gpt'], plove['new'])[0][0].cpu().detach().numpy()
print("KL for love", KLlovenew)

KLlove = np.zeros(len(checkpoints))
ind = 0 

for i in checkpoints:
  # print(i + ':')
  KL = torch.distributions.kl.kl_divergence(plove['gpt'], plove[i])[0][0]
  KLlove[ind] = KL.cpu().detach().numpy()
  ind += 1
  

pthe = {}
KLthe = np.zeros(len(checkpoints))
ind = 0
the_input = torch.tensor(tokenizer.encode("the", add_special_tokens=True)).unsqueeze(0)
the_input = the_input.to(device)

for i in checkpoints:
    
  pthe[i] = torch.distributions.categorical.Categorical(m(checkdict[i](the_input)[0]))
for i in ['gpt', 'final', 'new']:
  pthe[i] = torch.distributions.categorical.Categorical(m(checkdict[i](the_input)[0]))
for i in checkpoints:
  # print(i + ':')
  KL = torch.distributions.kl.kl_divergence(pthe['gpt'], pthe[i])[0][0]
  KLthe[ind] = KL.cpu().detach().numpy()
  ind += 1

KLthenew = torch.distributions.kl.kl_divergence(pthe['gpt'], pthe['new'])[0][0].cpu().detach().numpy()
print("KL for the", KLthenew)

### Calculating KL for deregulation
pdereg = {}
KLdereg = np.zeros(len(checkpoints))
ind = 0
dereg_input = torch.tensor(tokenizer.encode(" deregulation", add_special_tokens=True)).unsqueeze(0)
dereg_input = dereg_input.to(device)

for i in checkpoints:
    
  pdereg[i] = torch.distributions.categorical.Categorical(m(checkdict[i](dereg_input)[0]))
for i in ['gpt', 'final', 'new']:
  pdereg[i] = torch.distributions.categorical.Categorical(m(checkdict[i](dereg_input)[0]))

for i in checkpoints:
  print(i + ':')
  KL = torch.distributions.kl.kl_divergence(pdereg['gpt'], pdereg[i])[0][0]
  print(KL)
  KLdereg[ind] = KL.cpu().detach().numpy()
  ind += 1

KLderegnew = torch.distributions.kl.kl_divergence(pdereg['gpt'], pdereg['new'])[0][0].cpu().detach().numpy()
print("KL for deregulation", KLderegnew)

### Calculating KL for !
pexc = {}
KLexc = np.zeros(len(checkpoints))
ind = 0
exc_input = torch.tensor(tokenizer.encode("!", add_special_tokens=True)).unsqueeze(0)
exc_input = exc_input.to(device)
for i in checkpoints:
  pexc[i] = torch.distributions.categorical.Categorical(m(checkdict[i](exc_input)[0]))
for i in ['gpt', 'final', 'new']:
  pexc[i] = torch.distributions.categorical.Categorical(m(checkdict[i](exc_input)[0]))
for i in checkpoints:
  print(i + ':')
  KL = torch.distributions.kl.kl_divergence(pexc['gpt'], pexc[i])[0][0]
  print(KL)
  KLexc[ind] = KL.cpu().detach().numpy()
  ind += 1

KLexcnew = torch.distributions.kl.kl_divergence(pexc['gpt'], pexc['new'])[0][0].cpu().detach().numpy()
print("KL for !", KLexcnew)

import numpy as np

import matplotlib.pyplot as plt


checkno = np.array([checkno = np.array([500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000]) * 10])

loss = np.array([1.4465, 1.3898, 1.3757, 1.3303, 1.3323, 1.2391, 1.2671, 1.2348, 1.2786, 1.2569, 1.1911, 1.2154, 1.1853, 1.2104, 1.1965, 
                 1.1342, 1.1596, 1.1586, 1.1675, 1.1498, 1.1050, 1.0867, 1.1283, 1.1257, 1.1336, 1.1354, 1.0625, 1.0395, 1.0753, 1.0788, 
                 1.0821, 1.0371, 1.0417, 1.0475, 1.0464, 1.0827, 1.0405, 1.0213, 1.0125, 1.0274]
                )
lossno = np.arange(2000, 82000, 2000)

plt.figure()
plt.plot(checkno, KLexc, label='"!"', color="tab:blue")
plt.plot(80000, KLexcnew, color="tab:blue", marker="^", markersize=8)
plt.plot(checkno, KLlove, label='"love"', color="tab:orange")
plt.plot(80000, KLlovenew, color="tab:orange", marker="^", markersize=8)
plt.plot(checkno, KLthe, label='"the"', color="tab:green")
plt.plot(80000, KLthenew, color="tab:green", marker="^", markersize=8)
plt.plot(checkno, KLdereg, label='" deregulation"', color="tab:red")
plt.plot(80000, KLderegnew, color="tab:red", marker="^", markersize=8)
plt.legend()
plt.ylabel('KL Divergence')
plt.xlabel('Iteration')
plt.show()

plt.figure()
plt.plot(lossno, loss)
plt.xlabel('Iteration')
plt.ylabel('Loss')

## Prompt:
prompt = 
inputs = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
inputs = inputs.to(device)

## for model made from scratch
outputs = checkdict['new'].generate(inputs, max_length=300, do_sample=True, top_p=0.95, top_k=100, temperature=1.1)
tokenizer.decode(outputs[0].cpu().numpy())

## for fine tuned model
outputs = checkdict['final'].generate(inputs, max_length=300, do_sample=True, top_p=0.95, top_k=100, temperature=1.1)
tokenizer.decode(outputs[0].cpu().numpy())

### You can obtain outputs from different check points by - 
outputs = checkdict['final'].generate(inputs, max_length=300, do_sample=True, top_p=0.95, top_k=100, temperature=1.1)
tokenizer.decode(outputs[0].cpu().numpy())
