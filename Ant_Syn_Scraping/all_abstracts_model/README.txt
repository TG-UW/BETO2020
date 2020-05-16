This model was trained on the entire* fulltext_pOmOo..whatever corpus. Abstracts only.

*Note that some of the corpus apparently contains non-ascii characters,
so this run used a try/except clause around opening some of the 
journal json files. So it wasn't definitively the entire corpus, but hopefully most of it.

The call to create this model was:

def w2v_main():
    """
    Method to execute training of word2Word2Vec
    """
    corpus_path = '/gscratch/pfaendtner/dacj/nlp/fulltext_pOmOmOo/'
    jlist = os.listdir(corpus_path)
    jlist.remove('README.txt')
    jlist = [corpus_path + journal for journal in jlist]

    # creating the multiloader iterator object
    multi_j_loader = MultiLoader(jlist, years='all', retrieval_type='abstract')

    # calling Word2Vec in the same manner I did with a
    model = Word2Vec(multi_j_loader, min_count=10, workers=1, size=50, iter=10)
    os.chdir('/gscratch/pfaendtner/dacj/nlp/models/all_abstracts_model/')
    model.save('all_abstract_model.model')

w2v_main()
