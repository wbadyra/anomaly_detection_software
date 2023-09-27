##ABSTRACT##
The purpose of this work was to implement a solution for anomaly detection in car sensors. Based on
gathered datasets thesis was created, which assumes some level of dependence in parameters va-
lues in given moment of time. Multiple methods were implemented in order to explore this relationship.
Each of them was based on different approach in solving the problem. Methods can be divided into two
categories - clustering algorithms and machine learning models. First of them needed input data in low-
dimensional space in order to allow proper visualisation and boost their performance. Machine learning
models, which were implemented, are decision trees, k-means algorithm, logistic regression and support
vector machines. Collected datasets came from various car models, which differ in technical specifica-
tion. It caused a shift in data distribution between them, which resulted in worse performance of models
trained on multiple data files from different datasets. To solve this problem domain adaptation algorithms
were implemented, which enables proper data samples re-weighting, minimizing effect of its original
distribution. To properly evaluate assumed thesis multiple tests were conducted in different scenarios.
Different errors characteristics were defined, its value was altered and distinct number of sensors were
interfered. Acquired results allowed choosing the most effective clustering method, the best machine
learning model and analysis of each parameter influence on models behaviour. Both selected methods
are marked by high effectiveness and stability in all covered scenarios. Developed solution exploiting
domain adaptation methods showed only little results improvement in proper experiments terms.
