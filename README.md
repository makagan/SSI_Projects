# Machine Learning Across The Frontiers - SSI 2023 Projects

Each HEP frontier presents its own Big Data challenges, inviting the use of AI/ML to tackle them.
Here we choose three specific challenges, one from each of the Energy, Intensity, and Cosmic Frontiers, that can be tackled during the school by small project teams.

Each has a dataset associated with it, which can be either downloaded to your local (or remote) computing resource, or imported to Google colab.
Your team might then pick up one of the approaches described in the lectures, and try and apply it.
We provide a number of tutorial notebooks below, that introduce the datasets and provide some possible starting points for you.

On the last Thursday of the school, we will hear very short presentations from each project team in a common slide deck, and award various small prizes.

For maximum community value, project teams should plan to submit their project notebook back to this repo via a pull request, so everyone can benefit from their hard work. Fork this repo and get to work!

Have a look at the [`Getting Started`](https://github.com/makagan/SSI_Projects/blob/main/slides/GettingStarted.pdf) slides to get started with Github and Google Colab.

## The Challenges

**Energy Frontier:** here, the challenge is to develop ML models for LHC jets.
These could be for classification, or generative modeling.
We provide a dataset to explore that includes various boosted jets, including high-level jet features, jet-images, and per-particle features.
Many thanks to SSI lecturer Jennifer Ngadubia, from whose recent [`course`](https://github.com/jngadiub/ML_course_Pavia_23/blob/main/) the materials for this challenge are drawn!

**Cosmic Frontier:** here, the challenge is to develop methods for mapping Dark Matter in the Universe from weak lensing data, after exploring some related inverse problems using LSST-like imaging data. We provide suitable weak lensing datasets.
Many thanks to SSI Lecturer François Lanusse for the materials for this challenge, which are based on the materials used at the [Quarks2Cosmos conference](https://github.com/EiffL/Quarks2CosmosDataChallenge/tree/colab)!

**Intensity Frontier:** here, the challenge is to...
Many thanks to SSI Organizer Kazu Terao for the materials for this challenge!

## SSI2023 Project Prerequisites

Prerequisites for the course include basic knowledge of GitHub, Colab and python. It is thus required before the course to go through [these](https://github.com/makagan/SSI_Projects/blob/main/slides/GettingStarted.pdf) slides as well as the following two python basics notebooks:

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

## Tutorials

We've organized a variety of tutorial notebooks below, grouped by Frontier (after some more general tutorials you may find helpful).
Note that your project might well benefit from techniques you pick up by looking for tutorials _across the Frontiers..._

### General: Advanced Python

* Intro to Numpy: [`numpy_intro.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/python_advanced/numpy_intro.ipynb)
* Intro to Pandas: [`pandas_intro.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/python_advanced/pandas_intro.ipynb)
* Intro to Matplotlib: [`matplotlib_intro.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/python_advanced/matplotlib_intro.ipynb)

### General: Introduction to PyTorch

* Intro to PyTorch: [`pytorch_intro.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/pytorch_basics/pytorch_intro.ipynb) and [`pytorch_NeuralNetworks.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/pytorch_basics/pytorch_NeuralNetworks.ipynb)


### General: PyTorch Geometric (PyG)
* Intro to PyTorch Geometric: [`1.IntroToPyG.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/pytorch_geometric_intro/1.IntroToPyG.ipynb)
* Node classification with PyG on Cora citation dataset: [`2.KCNodeClassificationPyG.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/pytorch_geometric_intro/3.KCNodeClassificationPyG.ipynb)
* Graph classification with PyG on molecular prediction dataset: [`3.TUGraphClassification.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/pytorch_geometric_intro/3.TUGraphClassification.ipynb)

### Energy Frontier: Basic NN with Keras for LHC jet tagging task

* Introduction to dataset and tasks [slides: [GettingStarted.pdf](https://github.com/makagan/SSI_Projects/blob/main/slides/GettingStarted.pdf)]
* Dataset exploration: [`1.LHCJetDatasetExploration.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/jet_notebooks/1.LHCJetDatasetExploration.ipynb)
* MLP implementation with Keras: [`2.JetTaggingMLP.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/jet_notebooks/2.JetTaggingMLP.ipynb)
* Conv2D implementation with Keras: [`3.JetTaggingConv2D.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/jet_notebooks/3.JetTaggingConv2D.ipynb)
* Conv1D implementation with Keras: [`4.JetTaggingConv1D.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/jet_notebooks/4.JetTaggingConv1D.ipynb)


### Energy Frontier: RNN, GNN and Transformer implementations for  LHC jet tagging task

* GRU for LHC jet tagging task: [`5.JetTaggingRNN.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/neural-networks/5.JetTaggingRNN.ipynb)
* Graph classification with PyG on LHC jet dataset: [`6.JetTaggingGCN.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/jet_notebooks/6.JetTaggingGCN.ipynb)
* Transformer model for LHC jet tagging with tensorflow: [`7.JetTaggingTransformer.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/jet_notebooks/7.JetTaggingTransformer.ipynb)

### Energy Frontier: Anomaly Detection for LHC jets
* Anomaly detection for LHC jets with AE [`8.JetAnomalyDetectionAE.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/jet_notebooks/8.JetAnomalyDetectionAE.ipynb)
* Anomaly detection for LHC jets with VAE [`9.JetAnomalyDetectionVAE.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/jet_notebooks/9.JetAnomalyDetectionVAE.ipynb)

### Cosmic Frontier: Differentiable Forward Models, Generative Models, And Variational Inference
* [`1.PartI-DifferentiableForwardModel.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/cf_notebooks/1.PartI-DifferentiableForwardModel.ipynb.ipynb)
    - How to write a probabilistic forward model for galaxy images with Jax + TensorFlow Probability
    - How to optimize parameters of a Jax model
    - Write a forward model of ground-based galaxy images
* [`2.PartII-GenerativeModels.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/cf_notebooks/2.PartII-GenerativeModels.ipynb)
    - Write an Auto-Encoder in Jax+Haiku
    - Build a Normalizing Flow in Jax+Haiku+TensorFlow Probability
    - Bonus: Learn a prior by Denoising Score Matching
    - Build a generative model of galaxy morphology from Space-Based images
* [`3.PartIII-VariationalInference.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/cf_notebooks/3.PartIII-VariationalInference.ipynb)
    - Solve inverse problem by MAP
    - Learn how to sample from the posterior using Variational Inference
    - Bonus: Learn to sample with SDE
    - Recover high-resolution posterior images for HSC galaxies
    - Propose an inpainting model for masked regions in HSC galaxies
    - Bonus: Demonstrate single band deblending!

### Cosmic Frontier: Dark Matter Mass-Mapping using Real HSC Weak Gravitational Lensing Data
* Open challenge [`4.MappingDarkMatterDataChallenge.ipynb`](https://github.com/makagan/SSI_Projects/blob/main/cf_notebooks/4.MappingDarkMatterDataChallenge.ipynb)
    - Use Jax to write a differentiable model for weak gravitational lensing
    - Use an analytic Gaussian prior to solve the inverse problem (Wiener Filtering)
    - Use Denoising Score Matching to learn the score of a prior distribution
    - Use Stochastic Differential Equations for sampling from the posterior

## Other Resources

* Pattern Recognition and Machine Learning, Bishop (2006)
* Deep Learning, Goodfellow et al. (2016) -- [`link`](https://www.deeplearningbook.org/)
* Introduction to machine learning, Murray (2010) -- [`video lectures`](http://videolectures.net/bootcamp2010_murray_iml/)
* Stanford ML courses -- [`link`](https://ai.stanford.edu/stanford-ai-courses/)
