This is a project developed for CAS_NLP module 6. The app is doing the following:

1-	You enter how many days -> N
2-	The app queries the astroph.EP arXiv to download the abstracts from the N last days
3-	Then, I used a specially trained Bert-like model to compute the embeddings of all papers
4-	The embeddings are printed on the screen as heatmaps
5-	And finally, there is a download button to download the embeddings as a numpy file

The streamlit app is available here: https://2rbcfstcqkh6thpmernwja.streamlit.app/
