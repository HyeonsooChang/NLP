#!/usr/bin/env python
# coding: utf-8

# ### 텍스트 전처리(Text preprocessing)

# 01) 토큰화(Tokenization)
# - 주어진 코퍼스에서 토큰이라 불리는 단위로 나누는 작업을 토큰화
# 
# 1. 단어 토큰화(Word Tokenization)
# - 토큰의 기준을 단어로 하는 경우

# In[3]:


from nltk.tokenize import word_tokenize
print(word_tokenize("Don't be fooled by the dark sounding name,Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))


# Don't를 Do와 n't로 분리하였으며, 반면 Jone's는 Jone과 's로 분리한 것을 확인

# In[4]:


from nltk.tokenize import WordPunctTokenizer
print(WordPunctTokenizer().tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))


# WordPunctTokenizer는 구두점을 별도로 분류하는 특징

# In[5]:


from tensorflow.keras.preprocessing.text import text_to_word_sequence
print(text_to_word_sequence("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))


# 케라스의 text_to_word_sequence는 기본적으로 모든 알파벳을 소문자로 바꾸면서 구두점(마침표, 느낌표 등)을 제거한다. 그러나 어퍼스트로피는 보존한다.

# ### 토큰화에서 고려해야할 사항
# 
# 1) 구두점이나 특수 문자를 단순 제외해서는 안된다
# 2) 줄임말과 단어 내에 띄어쓰기가 있는 경우

# # 표준 토큰화 예제

# In[6]:


from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()
text = "Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."
print(tokenizer.tokenize(text))


# 규칙 1. 하이푼으로 구성된 단어는 하나로 유지한다.
# 규칙 2. doesn't와 같이 아포스트로피로 '접어'가 함께하는 단어는 분리해준다.

# ### 문장 토큰화(Sentence Tokenization)

# In[7]:


from nltk.tokenize import sent_tokenize
text="His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to make sure no one was near."
print(sent_tokenize(text))


# In[8]:


from nltk.tokenize import sent_tokenize
text="I am actively looking for Ph.D. students. and you are a Ph.D student."
print(sent_tokenize(text))


# In[9]:


import kss
text='딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어려워요. 농담아니에요. 이제 해보면 알걸요?'
print(kss.split_sentences(text))


# # 품사 태깅(Part-of-speech tagging)

# 품사에 따라 단어의 의미가 달라지기도 한다. 그에 따라 단어 토큰화 과정에서 각 단어가 어떤 품사로 쓰였는지를 구분해 놓는 걸을 품사 태깅이라고 한다.

# In[10]:


from nltk.tokenize import word_tokenize
text="I am actively looking for Ph.D. students. and you are a Ph.D. student."
print(word_tokenize(text))


# In[11]:


import nltk
nltk.download('averaged_perceptron_tagger')

from nltk.tag import pos_tag
x=word_tokenize(text)
pos_tag(x)


# In[ ]:




