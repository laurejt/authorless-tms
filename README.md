Authorless TMs
==============
This repository contains code and metadata to support work described in
Thompson and Mimno, "Authorless Topic Models: Biasing Models Away from
Known Structure" (COLING 2018).


Metadata
--------
The Science Fiction collection can be reconstructed from the metadate file scifi.tsv.
The title, author, and HathiTrust volume identifier is provided for each work in the corpus.
HathiTrust volume identifiers are uniquely tied to a single work within HathiTrust.
These identifiers can be used to obtain page-level features from the [HTRC Extracted 
Features Dataset](https://wiki.htrc.illinois.edu/display/COM/Extracted+Features+Dataset).

The U.S. State Supreme Courts collection can similarly be reconstructed from the 
metadata file state\_courts.tsv.
The state, date, and CourtListener Opinion ID is provided for each court opinion in the 
collection.
Opinion IDs can be used to obtain court opinion text via CourtListener's 
[REST](https://www.courtlistener.com/api/rest-info) or 
[Bulk Data](https://www.courlistener.com/api/bulk-info) APIs.


Input Corpus Format
-------------
The included programs require a specific input corpus format similar to Mallet's format.
A collection is represented by a single file with one document per line.
A document is represented by three tab-separated fields:

```
[document id]	[author label]	[text]
```

Document text is assumed to be preprocessed such that tokens are space-separated.
The program does not make any assumptions about the form of document ids or author labels,
but for compatibility with Mallet these fields should not contain whitespace.

The programs also require a file containing the working vocabulary for the corpus, with 
one word per line.
Any words within a document's text that are not in the vocabulary file will be ignored 
(i.e. stopped).

#### Constructing the input corpus
A correctly formatted input corpus can be constructed from a Mallet collection file 
generated with the import-file option. This allows users to incorporate Mallet's corpus 
preprocessing capabilities. The script get\_input.sh builds an input corpus from a 
Mallet collection file, optionally a minimum document length can be specified for the 
resulting corpus.
```
./get_input.sh [Mallet collection file] [output corpus file] ([min doc length])
```

#### Constructing a working vocabulary
A working vocabulary file can be constructed from a Mallet collection file generated with 
the import-file option. The script get\_vocab.sh builds a working vocabulary from a 
corpus's corresponding Mallet collection file.

```
./get_vocab.sh [Mallet collection file] [output vocab file]
```


Evaluating Topic-Author Correlation
-----------------------------------
Two separate programs are provided for measuring topic-metadata correlation of trained 
LDA topic model: topic\_author\_correlation.py and estimated\_author\_correlation.py.
These programs output a tab-separated file containing the author entropy, 
minus major author, and balanced author scores for each topic in the model, with one 
topic per line:
```
[topic number]	[author entropy]	[minus major author]	[balanced authors]
```

### topic\_author\_correlation.py
This program calculates the author-topic correlation measures for each topic in a 
Mallet LDA topic model as detailed in the paper. The measurements are constructed 
directly from the topic assignments stored within the Mallet state file.
```
python topic_author_correlation.py --input FILE --input-state FILE --vocab FILE --output FILE [-v] 
```
#### Flags

##### Required
--input FILE : The input corpus in the required format of one document per line,
                with tab-separated fields containing the corresponding document id,
                author label, and text.

--vocab FILE : File of the working vocabulary, with one word per line.

--input-state FILE : The Mallet state file containing the gzipped Gibb sampling state,
                     produced by the --output-state flag.


--output FILE : The location to save the topic-author correlation measurements.

##### Optional
-v, --verbose : Increases program verbosity, specifically producing more program progress 
                information.

### estimated\_topic\_author\_correlation.py
This program calculates the author-topic correlation measures for each topic in a 
Gensim or Mallet LDA topic model. This program constructs topic-token assignments 
using the trained model's document-topic distributions theta and topic-word 
distributions phi. 

This program requires a tool option which specifies whether the evaluated topic 
model was trained by mallet or gensim.

#### Gensim Option
```
python estimated_topic_author_correlation.py gensim --input FILE --lda-model FILE --output FILE [-v]
```

##### Required
--input FILE : The input corpus in the required format of one document per line,
                with tab-separated fields containing the corresponding document id,
                author label, and text.

--lda-model FILE : The file containing the saved gensim ldamodel.

--output FILE : The location to save the topic-author correlation measurements.

##### Optional
-v, --verbose : Increases program verbosity, specifically producing more program progress 
                information.


#### Mallet Option
```
python estimated_topic_author_correlation.py mallet --input FILE --vocab FILE --topic-word-weights FILE --doc-topics FILE --output FILE [-v]
```

##### Required
--input FILE : The input corpus in the required format of one document per line,
                with tab-separated fields containing the corresponding document id,
                author label, and text.

--vocab FILE : File of the working vocabulary, with one word per line.

--topic-word-weights FILE : Mallet-generated topic word weights file, produced by 
                            the --topic-word-weights-file flag.

--doc-topics FILE : Mallet-generated doc topics file produced by the 
                    --output-doc-topics flag.

--output FILE : The location to save the topic-author correlation measurements.

##### Optional

-v, --verbose : Increases program verbosity, specifically producing more program progress 
                information.


Reducing Topic-Author Correlation
---------------------------------
The program downsample\_corpus.py modifies a corpus using contextual probabilistic 
subsampling to reduce topic-author correlation in resulting topic models.
It outputs a downsampled version of the input corpus.

#### Flags

##### Required
--input FILE : The input corpus in the required format of one document per line,
                with tab-separated fields containing the corresponding document id,
                author label, and text.

--vocab FILE : File of the working vocabulary, with one word per line.

--output FILE : The location to save the downsampled corpus.

##### Optional
--threshold NUM : The probability threshold parameter for the subsampling algorithm.
                  This value should be set to a value between 0 and 1. Higher values 
                  will result in more aggressive token removal, while smaller values
                  will be more conservative. This parameter is set to 0.05 by default.
                  The default selected for its high performance in the paper's 
                  experiments, and should be modified with caution.

--min-doc-length N : Specifies the required minimum document length for the downsampled
                     topic. After subsampling, all docuemnts with lengths less than this
                     value will be removed. The default is 20. Because we're subsampling 
                     the documents may get much shorter and become noninformative for 
                     training.

--ignore-case : Makes text handling, for both the input corpus and working vocabulary,
                case-insensitive.

-v, --verbose : Increases program verbosity, specifically producing more program progress 
                information.


