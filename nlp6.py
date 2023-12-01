# Name: Raviraj Nehul
# Roll No: 41
# Batch: B3
# Pract no 6: Implement and visualize dependency parsing of textual input using Stanford and spacy library

import spacy
from spacy import displacy

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

text = "My name is Raviraj. I am from Sanjivani College of Engineering"
doc = nlp(text)

for token in doc:
    print(f"""
    TOKEN: {token.text}
    =====
    tag_ = {token.tag_}
    head.text = {token.head.text}
    dep_ = {token.dep_}"""
    )

# Use displacy to visualize the dependency tree
displacy.serve(doc, style="dep")


'''
    TOKEN: My
    =====
    tag_ = PRP$
    head.text = name
    dep_ = poss

    TOKEN: name
    =====
    tag_ = NN
    head.text = is
    dep_ = nsubj

    TOKEN: is
    =====
    tag_ = VBZ
    head.text = is
    dep_ = ROOT

    TOKEN: Raviraj
    =====
    tag_ = NNP
    head.text = is
    dep_ = attr

    TOKEN: .
    =====
    tag_ = .
    head.text = is
    dep_ = punct

    TOKEN: I
    =====
    tag_ = PRP
    head.text = am
    dep_ = nsubj

    TOKEN: am
    =====
    tag_ = VBP
    head.text = am
    dep_ = ROOT

    TOKEN: from
    =====
    tag_ = IN
    head.text = am
    dep_ = prep

    TOKEN: Sanjivani
    =====
    tag_ = NNP
    head.text = College
    dep_ = compound

    TOKEN: College
    =====
    tag_ = NNP
    head.text = from
    dep_ = pobj

    TOKEN: of
    =====
    tag_ = IN
    head.text = College
    dep_ = prep

    TOKEN: Engineering
    =====
    tag_ = NNP
    head.text = of
    dep_ = pobj

Using the 'dep' visualizer
Serving on http://0.0.0.0:5000 ...

'''