from nltk.tokenize import word_tokenize, RegexpTokenizer
import re

tweets = list()

with open('Health-Tweets/nbchealth.txt', 'r') as file:
    for line in file:
            line = line.split('|')[2]
            line = re.sub(r'http[\S]*\s','',line)

            tokenizer = RegexpTokenizer(r'\b(?<!@)\S+\b')
            line = tokenizer.tokenize(line.lower())

            tweets.append(line)
