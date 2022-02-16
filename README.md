 # scorEpochs - Python version
This is the Python version of the tool

Developed on Python 3.6

<br><br>

## Usage
This tool can be used through the command line (do not be afraid to put spaces, they will be automatically managed) or by importing it

In the last case you have two possibility: 
 - Import the function from the module:
 
  ```python
    from scorEpochs import scorEpochs 
    idx_best, epoch, scores = scorEpochs(cfg, data)
  ```
       
       
 - Import the module and use the function through the dot notation: 
 
  ```python
    import scorEpochs
    idx_best, epoch, scores = scorEpochs.scorEpochs(cfg, data)
  ```

<br>

The **data** parameter represents a 2D (channels x samples) matrix.

<br>

The **cfg** parameter is a dictionary which has the following keys:
- **freqRange**, in which the interested frequency band is defined as a list which contains the related cut frequencies
- **fs**, which represents the sampling frequency of the time series
- **windowL**, which identify the number of seconds of each epoch
- **smoothFactor**, which represents the window of the moving average filter which have to be applied on the power spectrum of each epoch (optional, the moving average is not computed if this parameter is omitted)

An example for this parameter is:
```python
cfg = {'freqRange':[1, 100], 'fs':500, 'windowL':20, 'smoothFactor':3}
```

<br>

[Click here to have a view of a demonstration on the usage of this tool.](https://colab.research.google.com/drive/1ygD5aMjdzpjMy6NyeaI47GX7Jg-jPvJv?usp=sharing)

<br><br>

## Required libraries
 - Numpy
 - Scipy
