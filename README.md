# Machine learning For Collider Physics Project for SSI 2023

This project will explore LHC Jet data sets and enable participants to
develop ML models for jets, such as for classification or generative
modeling. More specifically, the data includes various boosted jets,
including high-level jet features, jet-images, and per-particle
features. 


The project materials have been adapted from a recent [`course`](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/)
by Jennifer Ngaduiba.

## Prerequisites

Prerequisites for the course include basic knowledge of GitHub, Colab and python. It is thus required before the course to go through [these](https://github.com/makagan/SSI_Projects/blob/main/slides/GettingStarted.pdf) slides as well as the two python basics notebooks: 

* [`python_intro_part1.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/python_basics/python_intro_part1.ipynb)
    * Quickstart
    * Indentation
    * Comments
    * Variables
    * Conditions and `if` statements
    * Arrays
    * Strings
    * Loops: `while` and `for`
    * Dictionaries
* [`python_intro_part2.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/python_basics/python_intro_part2.ipynb)
    * Functions
    * Classes/Objects
    * Inheritance
    * Modules
    * JSON data format
    * Exception Handling
    * File Handling
 
## Advanced python
    * Intro to Numpy: [`numpy_intro.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/python_advanced/numpy_intro.ipynb)
    * Intro to Pandas: [`pandas_intro.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/python_advanced/pandas_intro.ipynb)
    * Intro to Matplotlib: [`matplotlib_intro.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/python_advanced/matplotlib_intro.ipynb)

## Machine Learning Tutorials

### Hands-on: basic NN with Keras for LHC jet tagging task
    * Introduction to dataset and tasks [slides: [3.LHCJetTaggingIntro.pdf](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/slides/3.LHCJetTaggingIntro.pdf)]
    * Dataset exploration: [`1.LHCJetDatasetExploration.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/jet_notebooks/1.LHCJetDatasetExploration.ipynb)
    * MLP implementation with Keras: [`2.JetTaggingMLP.ipynb`](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/neural-networks/2.JetTaggingMLP.ipynb)
    * Conv2D implementation with Keras: [`3.JetTaggingConv2D.ipynb`](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/neural-networks/3.JetTaggingConv2D.ipynb)
    * Conv1D implementation with Keras: [`4.JetTaggingConv1D.ipynb`](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/neural-networks/4.JetTaggingConv1D.ipynb)


### Hands-on: RNN and GNN implementations for different tasks
    * GRU for LHC jet tagging task: [`5.JetTaggingRNN.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/neural-networks/5.JetTaggingRNN.ipynb)
    * Intro to PyTorch: [`pytorch_intro.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/pytorch_basics/pytorch_intro.ipynb) and [`pytorch_NeuralNetworks.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/pytorch_basics/pytorch_NeuralNetworks.ipynb)
    * Intro to PyTorch Geometric (PyG): [`6.IntroToPyG.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/neural-networks/6.IntroToPyG.ipynb)
    * Node classification with PyG on Cora citation dataset: [`7.KCNodeClassificationPyG.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/neural-networks/7.KCNodeClassificationPyG.ipynb)
    * Graph classification with PyG on molecular prediction dataset: [`8.TUGraphClassification.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/neural-networks/8.TUGraphClassification.ipynb)
    * Graph classification with PyG on LHC jet dataset: [`9.JetTaggingGCN.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/jet_notebooks/9.JetTaggingGCN.ipynb)

 ### Hands-on:
    * Transformer model for LHC jet tagging with tensorflow: [`10.JetTaggingTransformer.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/jet_notebooks/10.JetTaggingTransformer.ipynb)


 ### Hands-on:
    * Generate data with vanilla GAN: [`11.VanillaGAN_FMNIST.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/jet_notebooks/11.VanillaGAN_FMNIST.ipynb)
    * Generate data with VAE: [`12.VAE_FMNIST.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/jet_notebooks/12.VAE_FMNIST.ipynb)
    * Anomaly detection for LHC jets with AE [`13.JetAnomalyDetectionAE.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/jet_notebooks/13.JetAnomalyDetectionAE.ipynb)
    * Anomaly detection for LHC jets with VAE [`14.JetAnomalyDetectionVAE.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/jet_notebooks/13.JetAnomalyDetectionVAE.ipynb)

## Resources

* Pattern Recognition and Machine Learning, Bishop (2006)
* Deep Learning, Goodfellow et al. (2016) -- [`link`](https://www.deeplearningbook.org/)
* Introduction to machine learning, Murray (2010) -- [`video lectures`](http://videolectures.net/bootcamp2010_murray_iml/)
* Stanford ML courses -- [`link`](https://ai.stanford.edu/stanford-ai-courses/)
