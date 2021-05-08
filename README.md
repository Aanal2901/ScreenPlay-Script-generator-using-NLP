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
##### All the models were trained on Kaggle notebook, setting accelerator in GPU. We preferred training on Kaggle because it was faster than colab. 
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
Oh my god, man. There were people driving. I think they were looking up, you know what we mean? A couple of people got over there. So those folks were driving over the highway.They're over'


#### Model Trained from scratch output:
Epochs - 10 Words - 300
I 've! They. You don’t got! Who, you! It was! I was! It was! You got! I think! Who. You know! You got! You are G! You didn’t You’ re This thought. To're You're is! What a great! It’'re P couldn do --K was a handsome...?! got! I have! I never You want am liked! I didn had a good real... an - an big guy know't got to Come and up! I know, and OK. What are You made a great-- was a kid a bit. You don!" was all was a liar, I didn! I didn't been, in is a big kid. You're MT saw to Three, in a GK ever. ( got to What took a lot! You knew in always! Who're all, you! I'm got got...M got to This took. How should got to decided a big sweet!?... you! You're You're married child have got to You're


#### Fine tuned
Epochs - 10 Words - 300  
to me, I think this is like the worst type of scandal or something and then you’re trying to get the information out. I guess it is sort of interesting, the people on the Internet do this all the time. When you're able to get information or a tip on people you're like "Where have you gotten it from?" Well, that I thought of at the time.And then we just go through it and put it through a series of filters and then I just get sort of sad because I know there’s nothing wrong at all. I know, as you know, that kind of people, that kind of people you know, they don't know what they’re talking about. We just go out there and read information on them. As soon as someone comes in they don't know what it is. And when they go out there and take a look, I can guarantee you, people tell you, what they’re actually doing is the truth. That’s the trick in that.
MARK : You're saying some people ask
