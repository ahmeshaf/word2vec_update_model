"""
TODO:

define a function which takes in inputs word2vec model, and input text.
Then update the model with extra vocabulary found in input text
return updated model
"""


import gensim
import logging

from numpy import array, zeros, ones, float32 as REAL


def update(model, data, num_sentences, mincount=3):
    """
    Add new words from new data to the existing model's vocabulary,
    generate for them random vectors in syn0 matrix.
    For existing words, increase their counts by their frequency in the new data.
    Generate new negative sampling matrix (syn1neg).
    Then, train the existing model with the new data.
    """
    added_count = 0

    logging.info("Extracting vocabulary from new data...")
    newmodel = gensim.models.Word2Vec(min_count=1, sample=0, hs=0)
    newmodel.build_vocab(data)

    logging.info("Merging vocabulary from new data...")
    sampleint = model.vocab[model.index2word[0]].sample_int
    words = 0
    newvectors = []
    newwords = []
    for word in newmodel.vocab:
        words += 1
        if word not in model.vocab:
            v = gensim.models.word2vec.Vocab()
            v.index = len(model.vocab)
            model.vocab[word] = v
            model.vocab[word].count = newmodel.vocab[word].count
            model.vocab[word].sample_int = sampleint
            model.index2word.append(word)

            random_vector = model.seeded_vector(model.index2word[v.index] + str(model.seed))
            newvectors.append(random_vector)

            added_count += 1
            newwords.append(word)
        else:
            model.vocab[word].count += newmodel.vocab[word].count
        if words % 1000 == 0:
            logging.info("Words processed: %s" % words)

    logging.info("added %d words into model from new data" % (added_count))

    logging.info("Adding new vectors...")
    alist = [row for row in model.syn0]
    for el in newvectors:
        alist.append(el)
    model.syn0 = array(alist)

    logging.info("Generating negative sampling matrix...")
    model.syn1neg = zeros((len(model.vocab), model.layer1_size), dtype=REAL)
    model.make_cum_table()

    model.neg_labels = zeros(model.negative + 1)
    model.neg_labels[0] = 1.

    model.syn0_lockf = ones(len(model.vocab), dtype=REAL)

    logging.info("Training with new data...")
    model.train(data, total_examples=num_sentences)

    return model


def update_model(model, text_file, sentence_length=56):
    sentences = gensim.models.word2vec.LineSentence(text_file, max_sentence_length=sentence_length)
    return update(model, sentences, get_num_sentences(text_file, sentence_length))


def get_num_sentences(text_file, sentence_length=56):
    with open(text_file, 'r') as content_file:
        content = content_file.read()
    all_words = [word for word in content.split()]
    return len(all_words)/sentence_length