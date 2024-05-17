## How both models work (roughly):
Both use the same idea of collecting data. The model provides a comprehensive list of keywords based on a seed keyword that I add directly to the data. The changes I made include reading the CSV subreddit data file and picking out the most frequent keywords from the CSV file using the keyword list that the model provides. It basically compares the words in the CSV file to the list the model provides. 

However, I will add the list of words provided by the model for each Issue, just so we have a reference of what the model is actually doing. The keywords for both models are turning out to be very similar because we are strictly basing it on the subreddit CSV file. 

## Seed Keywords
UK:
1. IsraelPalestine - Israel, Palestine
2. Healthcare UK - NHS
3. Taxation UK - Tax
4. Climate Change UK - Climate
5. Brexit - Brexit

US:
1. IsraelPalestine - Israel, Palestine
2. Healthcare US - 
3. Taxation US - Tax
4. Climate Change US - Climate
5. Immigration US - Immigration

## Data that the models are trained on 

Both resources accessible on github and google drive

1. Empath: Trained on reddit data
2. Word2Vec: Uses a specific pretrained model called sense2vec which is trained on 2015 reddit data




