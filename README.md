# LDA_Topic_Modelling
Topic Modelling with Latent Dirichlet Allocation LDA

LDA is an unsupervised generative model that assigns topic distributions to documents.
The model assumes that each document contains more than one topic. The topics are assigned to the documents depending on the words in each document which contribute to each topic. The number of topics in the corpus must be specified a priori. So there must be raughly an idea of the topics in the data set. Several topics may share the same words.

Latent means hidden as it generates the following hidden variables: 

1) Topic distribution over the documents (each document will have a discrete distribution over all topics)

2) Words distribution for each topic (each topic will have a discrete distribution over all words)


An important point to note: although I have named some topics in the example above, the model itself does not actually do any "naming" or classifying of topics. But by visually inspecting the top contributing words of a topic i.e. the discrete distribution over words for a topic, one can name the topics if necessary after training. We will show this more later.

There a several ways to implement LDA, however I will speak about collapsed gibbs sampling as I usually find this to be the easiest way to understand it.

The model initialises by assigning every word in every document to a random topic. Then, we iterate through each word, unassign it's current topic, decrement the topic count corpus wide and reassign the word to a new topic based on the local probability of topic assignemnts to the current document, and the global (corpus wide) probability of the word assignments to the current topic. This may be hard to understand in words, so the equations are below.

In progress...
