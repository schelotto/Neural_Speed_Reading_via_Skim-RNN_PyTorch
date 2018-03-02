
# Introduction  
This is a PyTorch implementation of [Neural Speed Reading via Skim-RNN](https://arxiv.org/pdf/1711.02085.pdf) published on ICLR 2018.
![Skim RNN](skim_rnn.png)

The imdb dataset is used by default and stored in the */data* folder.

# Usage
```
python main.py [arguments]
```

# Arguments
```
-h, help				help
-large_cell_size		size of the large LSTM
-small_cell_size		size ofthe small LSTM
```