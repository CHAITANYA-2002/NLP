# Hierarchical Contextualized Representation for Named Entity Recognition

Codes for the paper **Hierarchical Contextualized Representation for Named Entity Recognition** in AAAI 2020

## Requirement

	Python: 3.6 or higher.
	PyTorch 0.4.1 or higher.


## Usage

Prepare training data and word/label embeddings in [sample_data](sample_data).

In ***training*** status:
`python main.py --config demo.train.config`

In ***test*** status:
`python main.py --config demo.test.config`

The configuration file controls the network structure, I/O, training setting and hyperparameters. 

+ BERT embeddings

  We use the [tool](https://github.com/Adaxry/get_aligned_BERT_emb) to gennerate the BERT embedding for NER. 


#### Models 
pre-trained model used [lstmcrf.model](https://drive.google.com/drive/folders/1G3kN1WsPJDVk9FdVUtIdv7DXd55p3yv0?usp=sharing). 
