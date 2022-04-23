# Musical Key Classifier

# The data
I used the following data sources:
-  <a href='https://colinraffel.com/projects/lmd/'>The Lakh MIDI Dataset v0.1</a>
- <a href='https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification'>GTZAN dataset</a>
- <a href='https://github.com/GiantSteps/giantsteps-mtg-key-dataset'>giantsteps-mtg-key-dataset</a>
- <a href='https://github.com/GiantSteps/giantsteps-key-dataset'>giantsteps-key-dataset</a>
- <a href='https://github.com/hendriks73/directional_cnns'>directional_cnns</a>

# The model
I implemented a CNN very close to the architecture mentioned in this paper:  
- <a href='https://arxiv.org/abs/1808.05340'>Filip Korzeniowski and Gerhard Widmer, Genre-Agnostic Key Classification With Convolutional Neural Networks</a>

I also added spectral-focused layers at beginning of network as used in this paper:
- <a href='http://smc2019.uma.es/articles/P1/P1_07_SMC2019_paper.pdf'>Hendrik Schreiber and Meinard Müller, Musical Tempo and Key Estimation using Convolutional Neural Networks with Directional Filters</a>
