# ScreenPlay Script generator using NLP

### GPT2
This can be considered as an implementation and improvement on the [Generative Pretrained Transformer 2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).  
We implement this pre trained model using our seed scene and obtain the first 300 words. After this we fine tune this model using our dataset. Finally we train the entire model only on our dataset and obtain results.  
Observation: The GPT2 pre trained when subjected to fine tuning performs worse. This shows that model in itself has achieved some kind of maxima and hence trying to fine tune results in poor performance. However when we train our model from scratch it performs better than fine tuned model.  

### KL Divergence  
To quantitatively compare the language models, I chose to look at their Kullback-Leibler divergence
(or relative entropy), which is a measure of how different two probability distributions are. It can also
be informally thought of as an abstract distance between two distributions.  

### This project is divided in three parts.
##### All the models were trained on Kaggle notebook, setting accelerator in GPU. 
##### Language used is Python.
##### Libraries - Transformers, torch, numpy, matplotlib
The [first model](https://github.com/Aanal2901/ScreenPlay-Script-generator-using-NLP/blob/main/gtp2_pretrained_model.py) is Pretrained GPT2 model imported using Transformer libraries.    
The [second model](https://github.com/Aanal2901/ScreenPlay-Script-generator-using-NLP/blob/main/gpt2_fine_tune.py) is made from scratch by training only on the scripts from all.txt file.   
The [third model](https://github.com/Aanal2901/ScreenPlay-Script-generator-using-NLP/blob/main/gpt2_fine_tune.py) is made by fine-tuning the GPT2 model using scripts from all.txt.

## Results:
prompt = '''JOHNNY
Well because it was an out of state bank. Anyway, I was working as a busboy in a hotel, and she was sitting, drinking her coffee, and she was so beautiful, and I say hi to her. That’s how we met.
MARK
So, I mean, what's the interesting part?
JOHNNY
Well the interesting part is that
'''

#### Pre Trained model output:
Epochs - 10 Words - 300  
that you really don't know her at all. I think I had seen her a couple times, and the first one I heard from her was. I heard that from a pretty old friend, she was the one who was like that with us. We were driving on the highway down there. My friend was driving over there with his girlfriend, and she got into a little bit of a mess, and it was kind of a hot-and-straw kind of place. Like she was saying something about a really big truck. She was like, "Oh my God." And her friend stopped him. "You're all so hot!" It was like, "What? You wanna go over there with me?" So at that point, what kind of reaction were people gonna get?

MARK
Oh my god, man. There were people driving. I think they were looking up, you know what we mean? A couple of people got over there.
So those folks were driving over the highway. They're over'

#### Model Trained from scratch output:

