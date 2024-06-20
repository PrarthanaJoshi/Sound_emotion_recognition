SOUND EMOTION RECOGNITION


Prarthana Joshi 				08-01-2024




We import numpy as np for using linear algebra and import panda for processing data and CSV files (comma separated values).
We import the os module which provides a way to interact with the operating system , including the file and directory manipulation. 
for dirname, _, filenames in os.walk('/kaggle/input'): 
	In this line , ‘os.walk’ is a generator that generates file names in a directory tree by 	either walking top-down or bottom-up through the directory trees.
The function here takes a starting directory as an argument (‘/kaggle/input’ is the directory in this case)
It returns a tuple for each directory it encounters during the walk. The tuple contains :
	Dirname : the current directory being processed 
	Filenames : A list of filenames in the current directory
for filename in filenames:
	Loops through the list of filenames in the current directory.
Print(os.path.join(dirname,filename))
	Prints the full path of each file by joining the ‘dirname’ and ‘filename’ using 	‘os.path.join()’. This ensures that the correct path separator is used based on the 	operating system.
In summary , the code recursively walks through the directory tree starting from ‘/kaggle/input’ and prints the full path of each file found in the process. This can be useful for listing and processing files in a directory and its sub-directories.  



The snippet imports several python libraries and sets up the environment for working with audio data. 
1.‘import pandas as pd’ : Imports the pandas library , a powerful data manipulation and analysis library. It is often used for working with tabular data.
2.‘import numpy as np’ : Imports the NumPy library , which provides support for large , multidimensional arrays and matrices , along with mathematical functions to operate on these elements.
3.‘import os’ : Imports the os module , which provides a way to interact with the operating system , allowing you to perform tasks like navigating directories , working with files , etc.
4.‘import seaborn as sns’ : This line imports the seaborn library and assign it the alias ‘sns’. Seaborn is a data visualization library based on Matplotlib, providing a high-level interface for creating informative and attractive statistical graphics.
5.‘import matplotlib.pyplot as plt’ : This line imports the pyplot module from the Matplotlib library and assigns it the alias ‘plt’ . Matplotlib is a comprehensive 2D plotting library for Python , and pyplot provides a simple and convenient interface for creating various types of plots.
6.‘import librosa’ : This line imports the librosa library , which is commonly used for audio analysis in Python. Librosa provides functions for loading and analyzing audio files, extracting features and more.
7.‘import librosa.display’ : This line specifically imports the display module from librosa , which included functions for displaying audio-related visualizations.
8.‘from IPython.display import Audio’ : This line imports the Audio class from the IPython.display module. IPython is an interactive computing environment and the Audio class is used to embed audio content (such as audio playback) in the Jupyter Notebook environment.
9.‘import warnings
          Warnings.filterwarnings(‘ignore’) : These lines import the warnings module and configure it to Ignore warnings. This is often done to suppress unnecessary warnings that may be displayed during the execution of the code.











LOAD THE DATASET 





These lines initialize two empty lists , ‘paths’ and ‘label’ , which will be used to store file paths and corresponding labels, respectively


This line starts a loop using ‘os.walk’ to iterate through the directory specified by the path ‘/kaggle/input’ and its sub directories. The ‘os.walk’ function generates the file name in a directory tree.
‘dirname’ : The current directory being processed.
‘_’ : A placeholder for subdirectories (which are not used in this loop.
‘filenames’ : A list of filenames in the current directory.



For each filename , the full path is constructed using ‘os.path.join’ by combining the current directory (‘dirname’) and the filename. The resulting path is then added to the ‘path’ list.


Here, the code extracts the label from the filename. It assumes that the label is separated by underscores in the filename and that the label is the last part before the file extension. It splits the filename using the underscores and takes the last part (‘[-1]’). Then, it splits again using the period to remove the file extension, and the resulting label is stored in the variable ‘label’.


The lowercase version of the label is added to the ‘labels’ list. This step is often done to standardize labels and avoid cases sensitivity issues.


This conditional statement checks if the number of paths collected (‘len(paths)’) is equal to 2800. If so, it breaks out of the outer loop. This suggests that the code is designed to stop collecting paths and labels after reaching a specific dataset size (2800 samples). 


Finally , this line prints a message indicating that the dataset has been loaded.




’len(paths)’ refers to the length of the list ‘paths’ . The above line of code gives us the total number of paths loaded from the dataset. The data collected will be limited within 2800 to avoid unnecessary processing.



Paths[:5] returns a new list containing the first five elements from the original ‘paths’ list. Slicing to extract a subset of elements from a sequence which is most likely a list or a similar iterable. 


Labels[:5] in Python is similar to the previous explanation but applied to a different sequence , presumably a list or some other iterable containing elements like labels. It returns a new list containing the first five elements from the original ‘labels’ list.



This code creates a Dataframe ‘df’ with two columns : ‘speech’ and ‘label’. The ‘speech’ column contains the element from the ‘paths’ list, and the ‘label’ column contains the elements from the ‘labels’ list. The ‘head()’ method is used to display the first few rows of the Dataframe. Note that the lists ‘paths’ and ‘labels’ should be defined before running this code.





The ‘value_counts()’ method in Pandas is used to count the occurrences of unique values in a Series. In your case , it looks like you want to count the occurrences of each unique label in the ‘label’ column of the Dataframe ‘df’. This will print a Series where the index represents unique labels , and the values represent the number of occurrences of each label in the ‘label’ column of the Dataframe.

EXPLORATORY DATA ANALYSIS

sns.countplot(data=df, x='label')

The ‘sns.countplot()’ function in Seaborn is used to show the counts of observations in each category. In our case we want to create a count plot for the ‘label’ column in the Data Frame ‘df’. The code will display a count plot where each unique label in the ‘label’ column is represented on the x-axis , and the height of each bar indicates the count of occurrences for that label. The resulting plot gives a visual representation of the distribution of labels in your Data Frame.




We have defined two functions , ‘waveplot’ and ‘spectrogram’ , to generate a waveform plot and a spectrogram, respectively , using the Librosa library in Python. 



This line defines a function named ‘waveplot’ that takes three parameters - ‘data’ , ‘sr’(sampling rate) and ‘emotion’.



This line creates a new figure  for the plot using Matplotlib. The ‘figsize’ parameter sets the width and height of the figure in inches. In this case , it’s set to 10 inches in width and 4 inches in height.



This line sets the title of the plot to the value of the ‘emotion’ parameter. The ‘size; parameter sets the font size of the title to 20.



This line uses the ‘waveshow’ function from the Librosa library to create a waveform plot of the audio data. ‘data’ is the audio signal , and ‘sr’ is the sampling rate. The waveform represents the amplitude of the audio signal over time.



This line displays the plot. It’s necessary to call this function to actually visualize the waveform plot.





This line defines a function named ‘spectogram’ that takes three parameters - ‘data’ , ‘sr’ and ‘emotion’.



This line computes the Short-Time Fourier Transform (STFT) of the audio data using Librosa. The STFT represents how the frequency content of the signal changes over time and is commonly used to create spectograms.



This line converts the magnitude spectrogram ‘abs(x)’ to decibels using the ‘amplitude-to-db’ function from Librosa. This transformation is often applied to better visualize the dynamic range of the spectogram.



This line creates a new figure for the plot using Matplotlib. The ‘figsize’ parameter sets the width and height of the figure in inches. In this case , it’s set to 11 inches in width and 4 inches in height.



This line sets the title of the plot to the value of the ‘emotion’ parameter. The ‘size’ parameter sets the font size of the title to 20.



This line displays the spectogram using the ‘specshow’ function from Librosa. It takes the magnitude spectogram ‘xdb’ and sets the sampling rate (‘sr’) and the axes (x-axis as time and y-axis as hertz).





This line adds a colorbar to the plot , which helps interpret the colors in the spectrogram. The colorbar typically represents the intensity or magnitude of the spectrogram.





This line assigns the string ‘fear’ to the variable ‘emotion’. It represents the emotion for which you want to analyze the studio.



This line extracts the file path associated with the specified emotion from the DataFrame ‘df’. It uses boolean indexing to filter rows where the ‘label’ column is equal to the specified emotion (‘fear’), then extracts the ‘speech’ column , and finally converts it to a NumPy array. The ‘[0]’ at the end is used to get the first element from the resulting array , assuming there is at least one matching path.



This line loads the audio file specified by the ‘path’ variable using Librosa. It returns two  values : ‘data’ , which is the audio signal , and ‘sampling_rate’ , which is the number of samples per second in the audio. 



This line calls the ‘waveplot’ function you defined earlier , passing the audio data (‘data’), the sampling rate (‘sampling rate’) , and the emotion (‘fear’) as arguments. This function generates and displays a waveform plot for the audio.



This line calls the ‘spectogram’ function you defined earlier , passing the audio data (‘data’), the sampling rate(‘sampling_rate’), and the emotion (‘fear’) as arguments. This function generates and displays a spectogram for the audio.




Assuming you're using IPython or a Jupyter notebook environment, this line plays the audio file specified by the path variable. The Audio function is usually provided by IPython.display.


The very same function is used to show a waveplot and spectogram for all other emotions loaded in dataset I.e. Angry , Disgust , Neutral , Sad , Pleasant and Happy. The line by line explanation is an exact copy of the emotion ‘fear’.











FEATURE EXTRACTION 


The function ‘extract_mfcc’ loads an audio file , extracts 3 seconds of audio starting from 0.5 seconds, computes the MFCCs , takes the mean of each MFCC coefficient across time , and returns the resulting one-dimensional array of MFCC values. This can be useful in tasks such as audio feature extraction for machine learning models.



This function from librosa library takes the filename as an argument and has two optional parameters :
‘duration=3’ : Specifies the duration (in seconds) to load from the audio file . In this case , it loads 3 seconds of audio.
‘offset==0.5)’ : Specifies the starting point (in seconds) from which the audio should be loaded. In this case , it starts from 0.5 seconds into the audio file.
‘y, sr’ : The function returns two values :
1.‘y’ : NumPy array representing the audio signal
2.‘sr’ : Sampling rate of the audio signal.

Sampling rate : refers to the number of samples (or measurement) of the audio waveform taken per second. It is often measured in Hertz or Kilohertz. In digital audio, an analog signal is converted into a digital representation by sampling it at regular intervals. The sampling rate determines how many samples are taken per second , and it is a crucial parameter in representing the original analog signal accurately. For example , if we take a sampling rate of 44.1 kHz , it means that 44,100 measurements are taken every second to represent the continuous audio signal.



‘librosa.feature.mfcc(y=y, sr=sr, n=mfcc=40)’ : This function computes the Mel-frequency cepstral coefficients (MFCC) from an audio signal. It takes the audio signal (‘y’), sampling rate (‘sr’),  and the number of MFCCs to compute (‘n_mfcc=40’ in this case).
‘.T’ : Transposes the matrix of MFCCs. This is done because it’s a common convention in machine learning to have features along the columns and samples/time frames along the rows.
‘np.mean(…, axis=0)’ : Calculates the mean along axis 0, effectively computing the mean of each MFCC coefficient across time. This results in a one-dimensional array containing the mean MFCC values for each coefficient.


	
This line returns the computed MFCC values as a one-dimensional NumPy array.


MFCC ( Mel-frequency cepstral coefficients )- Coefficients that represent the short-term power spectrum of a sound signal. They are widely used in speech and audio processing applications , particularly in tasks like speech recognition and music analysis. The term “cepstral” refers to taking the inverse of the Fourier Transform of the logarithm of the estimated spectrum of a signal.
Here's a breakdown of the steps involved in computing MFCCs:
Frame the Signal: The audio signal is divided into short overlapping frames, typically around 20 to 40 milliseconds long. This is done to capture the spectral characteristics of the signal over time.
Apply a Window Function: A window function (e.g., Hamming window) is applied to each frame to minimize the impact of spectral leakage.
Compute the Fast Fourier Transform (FFT): The FFT is applied to each framed signal to convert it from the time domain to the frequency domain, obtaining a power spectrum.
Mel Filtering: The power spectrum is then passed through a set of Mel filters. These filters are spaced in a way that mimics the human ear's frequency response. The Mel scale is a perceptual scale of pitches that approximates the human ear's response to different frequencies.
Log Transformation: The logarithm of the filter bank energies is taken. This step is essential for approximating the human ear's sensitivity to different frequency ranges.
Discrete Cosine Transform (DCT): The coefficients resulting from the logarithmic transformation are then transformed using the DCT. The DCT helps decorrelate the filter bank energies and reduces redundancy.
Selecting Coefficients: The resulting DCT coefficients are then typically truncated, and a subset of them is selected to form the MFCC feature vector.




The line of code extract_mfcc(df['speech'][0]) is calling a function called extract_mfcc on the audio file specified by the value in the first row of the 'speech' column in a DataFrame named df. Let's break it down:

1. DataFrame Access:
   - df['speech']: This part is accessing the column named 'speech' in the DataFrame df. It assumes that this column contains references or paths to audio files.

2. Extracting File Path:
   - df['speech'][0]: This part is extracting the value in the first row of the 'speech' column. Assuming these values are file paths, it's retrieving the file path of the audio file in the first row.

3. Calling extract_mfcc function:
   - extract_mfcc(df['speech'][0]): This line is calling a function named extract_mfcc and passing the audio file's path as an argument. The purpose of this function, as explained in a previous response, is to load the audio file, compute the Mel-frequency cepstral coefficients (MFCCs), take the mean across time, and return a one-dimensional array of MFCC values.

In summary, this line of code is using the extract_mfcc function to extract MFCCs from the audio file specified in the first row of the 'speech' column in the DataFrame df. The resulting MFCC values can be utilized for various tasks, such as feature extraction for machine learning models or audio analysis.



It appears that ‘df’ is a pandas DataFrame, and there is a column named ‘speech’ in this DataFrame. The code is using the ‘apply’ function to apply a custom function ‘extract_mfcc’ to each element in the ‘speech’ column. The result is assigned to a variable named ‘X_mfcc’. Here’s a breakdown of the code :
1.‘df[‘speech’]’ : This extracts the ‘speech’ column from the DataFrame 	‘df’. It assumes that the ‘speech’ column audio data.
2. ‘.apply(lambda x: extract_mfcc(x))’ : The ‘apply’ function is used to 	apply a given function to each element in the ‘speech’ column. In 	this case , a lambda function is used to apply the ‘extract_mfcc’ 	function to each audio data element (‘x’) in the ‘speech’ column.
3. ‘extract_mfcc(x)’ : This suggests that there is a function called ‘extract_mfcc’ that takes an audio signal (‘x’) as input and presumably returns its corresponding Mel-frequency cepstral coefficients (MFCCs).
4.‘X_mfcc = …’ : The result of applying the ‘extract_mfcc’ function to each element in the ‘speech’ column is stored in the variable ‘X_mfcc’. After this line , ‘X_mfcc’ would likely be a pandas series where each element contains the MFCCs of the corresponding audio signal. 



It would print a NumPy array where each column represents the MFCCs for a specific time frame in the audio signal.









‘X_mfcc’ : It seems like ‘X_mfcc’ is some iterable containing elements related to MFCC data. The second line converts the newly created list ‘X’ into a NumPy array. X prints a NumPy array containing the same elements as ‘X_mfcc’. The last line prints the shape of the NumPy array ‘X’ . This gives you the dimensions of the array. 




This line of code is using the ‘np.expand_dims’ function from the NumPy library. This function is used to add a new axis to the array at the specified position. ‘X’ : This is the NumPy array you are working with.
‘np.expand_dims(X, -1)’ : The first argument is the array ‘X’ that you want to modify. The second argument ‘-1’ indicates the position where the new axis should be added. In this case , ‘-1’ means adding the new axis at the last dimension of the array.
‘X.shape’ : This line is checking the shape of the modified array ‘X’ after the ‘np.expand_dims’ operation.
Putting it together , the code is effectively adding a new dimension to the existing NumPy array ‘X’ at the last position and then checking the shape of the modified array. The specific use case for adding a new dimension can vary, but it is often done when working with Convolutional Neural Network (CNNs) or other machine learning models that require input data to have a certain number of dimensions. Adding a new dimension might be necessary to match the expected input shape of a model.








“From sklearn.preprocessing import OneHotExcoder” : This line imports the ‘OneHotEncoder’ class from the ‘sklearn.preprocessing’ module. 
“enc= OneHotEncoder()” : An instance of the ‘OneHotEncoder’ class is created. This instance will be used to perform one-hot encoding. 
“y = enc.fit_transform(df[[‘label’]]) : This select the ‘label’ column and transforms the data. The result (‘y’) is a sparse matrix containing the one-hot encoded representation of the ‘label’ column. 




The ‘toarray()’ method is used to convert the sparse matrix ‘y’ into a dense NumPy array. A dense array stores all elements including the zeros, in a contiguous block of memory. This transformation can be useful if you need a regular NumPy array for further analysis or if your downstream algorithms or libraries don’t support sparse matrix representations. After this line is executed, the variable ‘y’ will be a dense NumPy array containing the one-hot encoded representation of the ‘label’ column from your DataFrame. Each row in this array corresponds to a data point , and each column corresponds to a unique category in the ‘ label ’ column , with 1s indicating the presence of that category and 0s elsewhere.

‘y.shape’ : This expression is used to retrieve the shape of the NumPy array or matrix ‘y’ . In the context of machine learning , this often corresponds to the shape of the target variable or the output variable. The return output  will be a tuple representing the  dimensions of the array. Suppose it returns (2800,7) then it means that there are 2800 datapoints/rows and 7 columns/features/classes.

LSTM MODEL :



1. `from keras.models import Sequential`: This line imports the `Sequential` class from the `keras.models` module. `Sequential` is a Keras class that allows you to create models layer by layer.

2. `from keras.layers import Dense, LSTM, Dropout`: This line imports the `Dense`, `LSTM`, and `Dropout` layers from the `keras.layers` module. These are the building blocks of neural networks. `Dense` is a standard fully connected layer, `LSTM` is a Long Short-Term Memory layer, and `Dropout` is a regularization technique that helps prevent overfitting.

3. `model = Sequential([...])`: This line creates a new Sequential model and initializes it with a list of layers. The layers are defined within square brackets `[]`.

4. `LSTM(256, return_sequences=False, input_shape=(40,1))`: This line adds an LSTM layer to the model. It has 256 units, `return_sequences=False` means the output will be a single vector for each input sequence, and `input_shape=(40,1)` specifies the shape of input data: 40 time steps with 1 feature per step.

5. `Dropout(0.2)`: This line adds a Dropout layer with a dropout rate of 0.2. Dropout randomly sets a fraction of input units to 0 at each update during training, which helps prevent overfitting.

6. `Dense(128, activation='relu')`: This line adds a fully connected (Dense) layer with 128 units and ReLU (Rectified Linear Unit) activation function. ReLU is a commonly used activation function that introduces non-linearity to the model.

7. `Dropout(0.2)`: This line adds another Dropout layer with a dropout rate of 0.2.

8. `Dense(64, activation='relu')`: This line adds another fully connected (Dense) layer with 64 units and ReLU activation function.

9. `Dropout(0.2)`: This line adds another Dropout layer with a dropout rate of 0.2.

10. `Dense(7, activation='softmax')`: This line adds the output layer with 7 units and softmax activation function. Softmax is used for multi-class classification problems. It squashes the outputs to be between 0 and 1 and normalizes them so that the sum of the outputs is 1, which can be interpreted as probabilities.

11. `model.compile(...)`: This line compiles the model. It specifies the loss function (`categorical_crossentropy`), optimizer (`adam`), and metrics to monitor (`accuracy`).

12.`model.summary()`: This line prints a summary of the model architecture, showing the layers, output shapes, and number of parameters. It's a helpful tool for inspecting the structure of your model.





This line of code is training the model using the training data `X` and corresponding labels `y`.

- `X`: This represents the input data, typically a numpy array or a TensorFlow tensor, containing the features used for training the model.

- `y`: This represents the labels or target values corresponding to the input data `X`. It should have the same number of samples as `X`.

- `validation_split=0.2`: This parameter splits the training data into training and validation sets. Here, 20% of the training data will be used for validation during training, and the remaining 80% will be used for actual training.

- `epochs=50`: This parameter specifies the number of epochs for which the model will be trained. An epoch is one complete pass through the entire training dataset.

- `batch_size=64`: This parameter determines the number of samples per gradient update. In other words, it specifies the number of training examples utilized in one iteration. Here, the training data will be divided into batches of size 64 for training.

The `fit` method trains the model using the specified parameters and returns a `history` object, which contains information about the training process such as the loss and accuracy on the training and validation sets for each epoch. This object can be used for visualization and further analysis of the training process.

Data Plotting :



This block of code is plotting the training and validation accuracy over the epochs to visualize the performance of the model during training.

- `epochs = list(range(50))`: This line creates a list containing the numbers from 0 to 49, representing the epochs. This will serve as the x-axis values for the plot.

- `acc = history.history['accuracy']`: This line extracts the training accuracy values from the `history` object returned by the `fit` method. The `'accuracy'` key holds the training accuracy values recorded during each epoch.

- `val_acc = history.history['val_accuracy']`: This line extracts the validation accuracy values from the `history` object. The `'val_accuracy'` key holds the validation accuracy values recorded during each epoch.

- `plt.plot(epochs, acc, label='train accuracy')`: This line plots the training accuracy values against the epochs. It uses `epochs` as the x-axis values and `acc` as the y-axis values. The `label` parameter specifies the label for this line in the legend.

- `plt.plot(epochs, val_acc, label='val accuracy')`: This line plots the validation accuracy values against the epochs. It uses `epochs` as the x-axis values and `val_acc` as the y-axis values. The `label` parameter specifies the label for this line in the legend.

- `plt.xlabel('epochs')`: This line sets the label for the x-axis as 'epochs'.

- `plt.ylabel('accuracy')`: This line sets the label for the y-axis as 'accuracy'.

- `plt.legend()`: This line adds a legend to the plot, which helps in distinguishing between the lines representing training and validation accuracy.

- `plt.show()`: This line displays the plot with the specified configurations. It shows how both training and validation accuracies change over the epochs, providing insights into the performance and potential issues like overfitting or underfitting.


This block of code is similar to the previous one, but it's plotting the training and validation loss over the epochs. Here's the breakdown:

- `loss = history.history['loss']`: This line extracts the training loss values from the `history` object returned by the `fit` method. The `'loss'` key holds the training loss values recorded during each epoch.

- `val_loss = history.history['val_loss']`: This line extracts the validation loss values from the `history` object. The `'val_loss'` key holds the validation loss values recorded during each epoch.

- `plt.plot(epochs, loss, label='train loss')`: This line plots the training loss values against the epochs. It uses `epochs` as the x-axis values and `loss` as the y-axis values. The `label` parameter specifies the label for this line in the legend.

- `plt.plot(epochs, val_loss, label='val loss')`: This line plots the validation loss values against the epochs. It uses `epochs` as the x-axis values and `val_loss` as the y-axis values. The `label` parameter specifies the label for this line in the legend.

- `plt.xlabel('epochs')`: This line sets the label for the x-axis as 'epochs'.

- `plt.ylabel('loss')`: This line sets the label for the y-axis as 'loss'.

- `plt.legend()`: This line adds a legend to the plot, which helps in distinguishing between the lines representing training and validation loss.

- `plt.show()`: This line displays the plot with the specified configurations. It shows how both training and validation losses change over the epochs, providing insights into the training progress and potential issues like overfitting or underfitting.
