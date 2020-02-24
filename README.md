# LDA_Topic_Modelling
Topic Modelling with Latent Dirichlet Allocation LDA

LDA is an unsupervised generative model that assigns topic distributions to documents.
The model assumes that each document contains more than one topic. The topics are assigned to the documents depending on the words in each document which contribute to each topic. The number of topics in the corpus must be specified a priori. The model does not "name" the topics. It is up to us to assign the topics to the topics the model creates (by the discrete distribution over words for a topic. So there must be raughly an idea of the topics in the data set. Several topics may share the same words.

Latent means hidden as it generates the following hidden variables: 

1) Topic distribution over the documents (each document will have a discrete distribution over all topics)

2) Words distribution for each topic (each topic will have a discrete distribution over all words)

The model initialy assigns every word in every document to a random topic. Then the algorithm iterates through each word, unassigning the current topic, decrement the number of topics across the corpus, and assigning the word to a new topic based on the local probability of topic assignment to the current document and the global (corpus-wide) probability of word assignments to the current topic.

Work in progress...
