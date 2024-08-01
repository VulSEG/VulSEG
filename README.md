# VulSEG
This is an official implementation for the paper: "VulSEG: Enhanced Graph-Based Vulnerability Detection System with Advanced Text Embedding". 

In the field of software security, the detection of vulnerabilities in source code has become increasingly important. Traditional methods based on feature engineering and statistical models are inefficient when dealing with complex code structures and large-scale data, while deep learning approaches have shown significant potential. Many detection methods involve converting source code into images for analysis. Although scalable, convolutional neural networks (CNNs) often fail to fully comprehend the complex structure and semantic relationships in the code, resulting in inadequate capture of high-level semantic features, which affects the accuracy of detection. This study introduces an innovative vulnerability detection framework, VulSEG, which significantly improves detection accuracy while maintaining high scalability. We combine Program Dependence Graph (PDG), Control Flow Graph (CFG), and Context Dependence Graph (CDG) to create a context-enhanced graph representation. Additionally, we have developed a composite feature encoding strategy that integrates Abstract Syntax Tree (AST) encoding with deep semantic security coding(Word2Vec+CSW-TF-IDF) to enhance understanding of code complexity and accuracy in predicting potential vulnerabilities. By incorporating the TextCNN and BiLSTM models, we further enhance feature extraction and long-sequence dependency handling capabilities.

Experimental results show that VulSEG outperforms nine state-of-the-art vulnerability detectors (namely RATS, FlawFinder, Vul-CNN, VulDeePecker, Devign, VGDetector,  DeepWukong, AMPLE and DeepVulSeeker) on the SARD dataset, achieving an 11.8% improvement in accuracy compared to other image-based methods.

## Usage
Step 1: Code preprocessing

*   Code normalization

    ```python
    python preprocessing.py -i ./data
    ```
*   Generate training corpus

    ```python
    python generate_corpus.py
    ```
*   Generate Word2Vec model

    ```python
    python generate_word2vec.py
    ```
*   Generate TF-IDF scores

    ```python
    python get_tfidf_dic.py.py
    ```
*   Generate infocode encoding

    ```python
    python get_infocode.py -i ./data/pdgs/Vul
    python get_infocode.py -i ./data/pdgs/No-Vul
    ```
*   Improved TF-IDF

    ```python
    python srw_score.py.py
    ```

Step 2: Generate pdg and cfg with joern's help

```python
# first generate .bin files
python gen_graph.py -i ./data/Vul -o ./data/bins/Vul -t parse
python gen_graph.py -i ./data/No-Vul -o ./data/bins/No-Vul -t parse

# then generate pdgs (.dot files)
python gen_graph.py -i ./data/bins/Vul -o ./data/pdgs/Vul -t export -r pdg
python gen_graph.py -i ./data/bins/Vul -o ./data/pdgs/No-Vul -t export -r pdg

# then generate cfgs (.dot files)
python gen_graph.py -i ./data/bins/Vul -o ./data/cfgs/Vul -t export -r cfg
python gen_graph.py -i ./data/bins/Vul -o ./data/cfgs/No-Vul -t export -r cfg
```

Step 3: Generate the image

```python
python img.py
```

Step 4: Generate data

```python
python generate_data.py -i ./data/outputs -o ./data/pkl -n 5
```

Step 5: Train

```python
python VulSEG.py -i ./data/pkl
```
## Acknowledgment
**We are especially grateful for the source code provided by [VulCNN](https://github.com/CGCL-codes/VulCNN). Our work is an improvement and extension on the basis of VulCNN.**
# VulSEG
# VulSEG
