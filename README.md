# Quantum Natural Language Processing : NEASQC WP 6.1

## Pre-setup
The following guide will walk you thorugh how to use our models. Prior to following those steps, you should ensure that you:
- Have a copy of the repository on your local machine.
- Have `python 3.10` installed on your local machine. Our models *might* turn out to be compatible with later versions of python but they were designed with and intended for 3.10.
- Have `poetry` installed on your local machine. You can follow the instructions on <a href="https://python-poetry.org/docs/#installation">the official website</a>.

## Setup
1. Position yourself in the `dev` branch.
2. Position yourself in the root of the repository where the files <code>pyproject.toml</code> and <code>poetry.lock</code> are located.
3. Run the command: <pre><code>$ poetry install</pre></code>. If you also want to install the dependencies to build `sphinx` documentation, use the command <pre><code>poetry install --with docs</pre></code> instead.
4. Activate the `poetry` usibg the command: <pre><code>poetry shell</code></pre>. More details can be found <a href="https://python-poetry.org/docs/basic-usage/#using-your-virtual-environment">here</a>.


## Running models
In this section we present our main models - Alpha 3, Beta 2 and Beta 3. 

Note that each model takes in a different input, but all produce the same output: 
* A `JSON` file containing the details of all the runs (loss, accuracy, runtime, etc.)
* A `.pt` file for each run with the final weights of the model at the end of the run.



### Alpha 3 

Alpha 3 follows a dressed quantum circuit (DQC) architecture, meaning that it combines a classical network architecture with a quantum circuit. A fully-connected quantum circuit is sandwiched between multi-layer perceptrons (MLPs). This model performs multiclass classification of natural language data.

The first MLP takes in sentence embeddings of dimension N and reduces them to an output of dimension Q ( < N ) where Q is the number of qubits of the circuit.

The second MLP takes the output of the quantum circuit as input (a vector of dimension Q), and outputs a vector of dimension C, where C is the number of classes. The final prediction of the class is made from this vector.

#### Files

There are two slight variations of the Alpha 3 model in this repository:

* The **standard** version, *alpha_3_multiclass_tests*. This model trains on a training dataset, and upon training, evaluates on a test dataset.

* The **cross-validation** version, *alpha_3_multiclass*, used internally for preliminary experiments. This model reads a dataset that in which sentences are labelled with their corresponding split. For each split S, the model takes all other sentences as training data, and uses the split S as validation data. This allows one to perform k-fold cross-validation on the model. Finally, the trained and validated model is evaluated on a test dataset.

The file defining the model and the flow of data for both versions is [alpha_3_multiclass_model.py](/neasqc_wp61/models/quantum/alpha/module/alpha_3_multiclass_model.py). 

Each version then has its own trainer file, which defines the training mechanism of the model. The trainer for the standard version is [alpha_3_multiclass_trainer_tests.py](/neasqc_wp61/models/quantum/alpha/module/alpha_3_multiclass_trainer_tests.py) and that of the cross-validation version is [alpha_3_multiclass_trainer.py](/neasqc_wp61/models/quantum/alpha/module/alpha_3_multiclass_trainer.py).

Each version also has a pipeline file, which pieces the model and trainer together and prcoesses the input and output. The pipeline file for the standard version is [use_alpha_3_multiclass_tests.py](/neasqc_wp61/data/data_processing/use_alpha_3_multiclass_tests.py) and that of the cross-validation version is [use_alpha_3_multiclass.py](/neasqc_wp61/data/data_processing/use_alpha_3_multiclass.py).

#### Datasets (standard version)

**NOTE: we recommend that you store all datasets in neasqc_wp61/data/datasets**

To run Alpha 3, you must have a dataset in CSV format consisting of 3 columns:
 
* 'class' - this column will contain the numbers that represents the class of each sentence (e.g. in binary classification, this could be 0 for a negative sentence, and 1 for a positive one). The numbers should be in the range [0, C-1] where C is the total number of classes.

* 'sentence' - this column will contain the natural language sentences that will be classified by the model.

* 'sentence_embedding' - this column will contain the sentence embeddings (e.g. BERT, ember-v1, etc.) corresponding to each sentence. The embeddings should be in standard list/vector notation format, e.g. [a,b,...,z].

If you have a CSV file with 'class' and 'sentence' labels, and you want to add a column with the corresponding BERT embeddings, you may use our [dataset_vectoriser.py](/neasqc_wp61/data/data_processing/dataset_vectoriser.py) script. From the root of the repo do:
```
cd neasqc_wp61/data/data_processing/
```
and then run the script:
```
python dataset_vectoriser.py <path-to-your-csv-dataset> -e sentence
```
This will produce a new CSV file identical to your dataset but with an additional column 'sentence_embedding' containing the embeddings for each sentence. This file will be saved to the same directory in which your dataset lives. You want to do this both for your train and test datasets.

**NOTE: the datasets that we used can be found [here](https://github.com/tilde-nlp/neasqc_wp6_qnlp/tree/v2-classical-nlp-models/neasqc_wp61/data/datasets). Please note that to use these yourself you will have to add the 'class' and 'sentence' column headers and convert the format to csv. This is very easy to do with the pandas Python library**

#### Datasets (cross-validation version)

If you wish to use our cross-validation version of the Alpha 3 model, simply ensure that your dataset (containing the training and validation data) has an additional column:
* 'split' - this column contains numbers that indicate the split to which the sentence belongs to. For K-fold cross-validation, these numbers should be in the range [0, K-1]

Adding this column is simple using the <code>pandas</code> Python library. Make sure you choose an appropriate number of splits based on the size of your dataset.

Once this column is present you can run the [dataset_vectoriser.py](/neasqc_wp61/data/data_processing/dataset_vectoriser.py) script as per above.

Your test dataset file does not require this 'split' column, only the 3 columns indicated in the previous section.

#### Running the model

The model has a number of parameters that must be specified through flags in the command line. These are:

* -s : an integer seed for result replication.
* -i : the number of iterations (epochs) for the training of the model.
* -r : the number of runs of the model (each run will be initialised with a different seed determined by the -s parameter).
* -u : the number of qubits of the fully-connected quantum circuit
* -d : q_delta, i.e. the initial spread of the quantum parameters (we recommend setting this to 0.01 initially).
* -p : the <code>PyTorch</code> optimiser of choice.
* -b : the batch size.
* -l: the learning rate for the optimiser.
* -w : the weight decay (this can be set to 0).
* -z : the step size for the learning rate scheduler.
* -g : the gamma for the learning rate scheduler.
* -o : path for the output file.

Additionally, only for the standard version, we have:

* -t : the path to the training dataset.
* -v : the path to the test dataset.

And, instead, for the cross-validation version we have:

* -f : path to the dataset containing the training and validation data with the split information as detailed before.
* -v : path to the test dataset.

Below we give an example on how to run both these versions from the command line. Make sure your Python environment is active and that you run this from the the *neasqc_wp61* directory where the `6_Classify_With_Quantum_Model.sh` is located.

**Standard Version**
```
bash 6_Classify_With_Quantum_Model.sh -m alpha_3_multiclass_tests -t <path to train data>  -v <path to test data> -p Adam -s 42 -r 1 -i 10 -u 4 -d 0.01 -b 2048 -l 0.002 -w 0 -z 150 -g 1 -o ./benchmarking/results/raw/
```
**Cross-Validation Version**
```
bash 6_Classify_With_Quantum_Model.sh -m alpha_3_multiclass_tests -f <path to split train and validation data>  -v <path to test data> -p Adam -s 42 -r 1 -i 10 -u 4 -d 0.01 -b 2048 -l 0.002 -w 0 -z 150 -g 1 -o ./benchmarking/results/raw/
```

### Beta 2 

#### Description

Beta 2 follows what we have called a semi-dressed quantum circuit (SDQC) architecture. The difference between a DQC and an SDQC is that the first MLP before the quantum circuit is removed. 

In Beta 2, the input to the circuit is a PCA-reduced sentence embedding (PCA = principal component analysis), i.e. a vector of size Q where Q is the number of qubits in the quantum circuit (Q is fixed to 8 in our code). One starts with a sentence embedding (e.g. BERT) of size N, reduces its dimension to Q using a PCA, and this resulting vector is plugged into the quantum circuit (with some rescaling).

This model is thus more reliant on the quantum circuit to make predictions than Alpha 3.

#### Files

Just like Alpha 3, Beta 2 consists of a standard and a cross-validation version.

The Beta 2 model architecture is defined in the [beta_2_3_model.py](/neasqc_wp61/models/quantum/beta_2_3/beta_2_3_model.py) file, along with that of the Beta 3 model (see next section for details).

The trainer files for Beta 2 are:
*  [beta_2_3_trainer_tests.py](/neasqc_wp61/models/quantum/beta_2_3/beta_2_3_trainer_tests.py) for the **standard** version.
* [beta_2_3_trainer.py](/neasqc_wp61/models/quantum/beta_2_3/beta_2_3_trainer.py) for the **cross-validation** version.

(These files also contain the training pipeline for Beta 3, more details in the next section)

The pipeline files for Beta 2 are:
* [use_beta_2_3_tests.py](/neasqc_wp61/data/data_processing/use_beta_2_3_tests.py) for the **standard** version.
* [use_beta_2_3.py](/neasqc_wp61/data/data_processing/use_beta_2_3.py) for the **cross-validation** version.

(Once again, these files are shared with Beta 3 given the similarities in architecture and data flow between the two models, More details on Beta 3 in the next section)

#### Datasets (standard version)

To run Beta 2, you must have a dataset in CSV format consisting of 4 columns:
 
* 'class' - this column will contain the numbers that represents the class of each sentence (e.g. in binary classification, this could be 0 for a negative sentence, and 1 for a positive one). The numbers should be in the range [0, C-1] where C is the total number of classes.

* 'sentence' - this column will contain the natural language sentences that will be classified by the model.

* 'sentence_embedding' - this column will contain the sentence embeddings (e.g. BERT, ember-v1, etc.) corresponding to each sentence. The embeddings should be in standard list/vector notation format, e.g. [a,b,...,z].

* 'reduced_embedding' - this column will contain the PCA-reduced sentence embeddings, in the same format as the full sentence embeddings.

Assuming you have a dataset with the first three columns (from following the instructions for Alpha 3), you can generate a new dataset with the additional 'reduced_embedding' column by using our [generate_pca_test_dataset.py](/neasqc_wp61/data/data_processing/generate_pca_test_dataset.py) script. Simply open the script and change the path in line 5 to that of your dataset, and the path in line 18 to your desired output name and directory. Save and close. From the root of the repo do:
```
cd neasqc_wp61/data/data_processing/
```
and then run the script:
```
python generate_pca_test_dataset.py
```
This will produce a new CSV file with the additional 'reduced_embedding' column. Make sure to do this both for your traing and testing datasets. 

#### Datasets (cross-validation version)

If you wish to use our cross-validation version of the Beta 2 model, first ensure that your dataset (with the train and validation data) has an additional column:

* 'split' - this column contains numbers that indicate the split to which the sentence belongs to. For K-fold cross-validation, these numbers should be in the range [0, K-1]

Adding this column is simple using the <code>pandas</code> Python library. Make sure you choose an appropriate number of splits based on the size of your dataset.

Once this is done, you need an addtional set of columns 'reduced_embedding_i'. These columns contain the PCA-reduced embeddings, with i indicating that the embeddings have been reduced with a PCA that has been fitted on the training data for split i (that is, all other splits != i). If you have a dataset with all other columns, these columns are easy to add using our [generate_pca_dataset.py](/neasqc_wp61/data/data_processing/generate_pca_dataset.py) script. 

Simply open the script, edit line 5 to include the path to your dataset containing the train+validation data, and edit line 30 with your desired output file path and name. Then save and close. From the root of the repo do:
```
cd neasqc_wp61/data/data_processing/
```
and then run the script:
```
python generate_pca_dataset.py 
```
This will produce a CSV file in the desired output path with the required format and columns.

For the test dataset, you do not need the 'split' columns, and you can use the [generate_pca_test_dataset.py](/neasqc_wp61/data/data_processing/generate_pca_test_dataset.py) script, which is described in the previous section, to reduce the emebddings in the 'sentence_embedding' column and add them to a new 'reduced_embedding' column.

#### Running the model

The model has a number of parameters that must be specified through flags in the command line. These are:

* -s : an integer seed for result replication.
* -i : the number of iterations (epochs) for the training of the model.
* -r : the number of runs of the model (each run will be initialised with a different seed determined by the -s parameter).
* -u : the number of qubits of the fully-connected quantum circuit
* -d : q_delta, i.e. the initial spread of the quantum parameters (we recommend setting this to 0.01 initially).
* -p : the <code>PyTorch</code> optimiser of choice.
* -b : the batch size.
* -l: the learning rate for the optimiser.
* -w : the weight decay (this can be set to 0).
* -z : the step size for the learning rate scheduler.
* -g : the gamma for the learning rate scheduler.
* -o : path for the output file.

* -f : the path to the training dataset (in the case of the **standard version**) or to the dataset containing the training and validation data (in the case of the **cross-validation version**).
* -v : the path to the test dataset.

Below we give an example on how to run both versions of Beta 2 from the command line. Make sure your Python environment is active and that you run this from the the *neasqc_wp61* directory.

**Standard Version**
```
bash 6_Classify_With_Quantum_Model.sh -m beta_2_tests -f <path to train dataset> -v <path to test dataset> -p Adam -s 42 -r 1 -i 10 -u 8 -d 0.01 -b 2048 -l 0.002 -w 0 -z 150 -g 1 -o ./benchmarking/results/raw
```
**Cross-Validation Version**
```
bash 6_Classify_With_Quantum_Model.sh -m beta_2 -f <path to split train and validation data>  -v <path to test data> -p Adam -s 42 -r 1 -i 10 -u 8 -d 0.01 -b 2048 -l 0.002 -w 0 -z 150 -g 1 -o ./benchmarking/results/raw
```

### Beta 3

#### Description

Beta 3 also follows a semi-dressed quantum circuit (SDQC) architecture. 

Unlike in Beta 2, our input to the quantum circuit is not a PCA-reduced sentence embedding, but rather a reduced fastText embedding vector of dimension Q (with some rescaling), where Q is the number of qubits of the circuit. Q is fixed to 8 in our code.

There is no need to use PCA to reduce the sentence embedding, as the fastText library conveniently includes a feature that allows us to reduce the fastText model to our desired dimension (dim=8 in the case of Beta 3).

#### Files

As you may have seen in the previous section on Beta 2, Beta 2 and 3 share model, trainer and pipeline files. This is due to their similarity in that the input to the quantum circuit is a reduced sentence embedding vector of dimension 8. The differences are mainly in the processinf of data, which is done separately for both models using if statements.

Just like Alpha 3 and Beta 2, Beta 3 has a **standard version** and a **cross-validation version**.

The Beta 3 model architecture is defined in the [beta_2_3_model.py](/neasqc_wp61/models/quantum/beta_2_3/beta_2_3_model.py) file.

The trainer files for Beta 3 are:
*  [beta_2_3_trainer_tests.py](/neasqc_wp61/models/quantum/beta_2_3/beta_2_3_trainer_tests.py) for the **standard** version.
* [beta_2_3_trainer.py](/neasqc_wp61/models/quantum/beta_2_3/beta_2_3_trainer.py) for the **cross-validation** version.

The pipeline files for Beta 3 are:
* [use_beta_2_3_tests.py](/neasqc_wp61/data/data_processing/use_beta_2_3_tests.py) for the **standard** version.
* [use_beta_2_3.py](/neasqc_wp61/data/data_processing/use_beta_2_3.py) for the **cross-validation** version.

#### Datasets (standard version)

To run Beta 2, you must have a dataset in CSV format consisting of 3 columns:
 
* 'class' - this column will contain the numbers that represents the class of each sentence (e.g. in binary classification, this could be 0 for a negative sentence, and 1 for a positive one). The numbers should be in the range [0, C-1] where C is the total number of classes.

* 'sentence' - this column will contain the natural language sentences that will be classified by the model.

* 'reduced_embedding' - this column will contain the reduced fastText embeddings in standard vector format, i.e. [a,b,...,z]

Assuming you have a CSV file with the first two columns, if you want to generate the reduced fastText embeddings and add them to a new 'reduced_embedding' column, you can use our [generate_fasttext_dataset.py](/neasqc_wp61/data/data_processing/generate_fasttext_dataset.py) script. Simply open the file, edit line 7 with the path to your CSV file, edit line 25 with your desired output path, then save and close. From the root of the repo do:
```
cd neasqc_wp61/data/data_processing/
```
and then run the script:
```
python generate_fasttext_dataset.py
```
which will generate a new CSV file with the fastText embeddings in the 'reduced_embedding' column. Make sure to do this both for your training and testing datasets.

#### Datasets (cross-validation version)

For the cross-validation version, you want the same columns as above, plus the following:
* 'split' - this column contains numbers in the range [0, K-1] where K is the number of folds in the cross-validation proceedure. This number will indicate what split the data belongs to.

If you have a dataset with the 'class', 'split' and 'sentence' column, and want to vectorise the sentences using fastText and add the result embeddings in a new 'reduced_embedding' column, you can use [generate_fasttext_dataset.py](/neasqc_wp61/data/data_processing/generate_fasttext_dataset.py) as described in the previous subsection.

#### Running the model

The model has a number of parameters that must be specified through flags in the command line. These are:

* -s : an integer seed for result replication.
* -i : the number of iterations (epochs) for the training of the model.
* -r : the number of runs of the model (each run will be initialised with a different seed determined by the -s parameter).
* -u : the number of qubits of the fully-connected quantum circuit
* -d : q_delta, i.e. the initial spread of the quantum parameters (we recommend setting this to 0.01 initially).
* -p : the <code>PyTorch</code> optimiser of choice.
* -b : the batch size.
* -l: the learning rate for the optimiser.
* -w : the weight decay (this can be set to 0).
* -z : the step size for the learning rate scheduler.
* -g : the gamma for the learning rate scheduler.
* -o : path for the output file.

* -f : the path to the training dataset (in the case of the **standard version**) or to the dataset containing the training and validation data (in the case of the **cross-validation version**).
* -v : the path to the test dataset.

Below we give an example on how to run both versions of Beta 3 from the command line. Make sure your Python environment is active and that you run this from the the *neasqc_wp61* directory.

**Standard Version**
```
bash 6_Classify_With_Quantum_Model.sh -m beta_3_tests -f <path to train dataset> -v <path to test dataset> -p Adam -s 42 -r 1 -i 10 -u 8 -d 0.01 -b 2048 -l 0.002 -w 0 -z 150 -g 1 -o ./benchmarking/results/raw
```
**Cross-Validation Version**
```
bash 6_Classify_With_Quantum_Model.sh -m beta_3 -f <path to split train and validation data>  -v <path to test data> -p Adam -s 42 -r 1 -i 10 -u 8 -d 0.01 -b 2048 -l 0.002 -w 0 -z 150 -g 1 -o ./benchmarking/results/raw
```