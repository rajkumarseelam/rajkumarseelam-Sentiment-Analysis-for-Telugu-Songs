import os
import codecs
import string
from gensim.models import FastText
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle

def preprocess(songs_list):
	telugu_characters  = [chr(i) for i in range(ord(b'\\u0c00'.decode('unicode_escape')), ord(b'\\u0c7f'.decode('unicode_escape'))+1)]
	updated_songs_list = []
	for song in songs_list:
		processed_song = ''
		for ch in song:
			if(ch in telugu_characters):
				processed_song += ch
			elif(ch == ' ' or ch == '.' or ch==','):
				if(processed_song[-1]!=' '):
					processed_song += ' '
		updated_songs_list.append(processed_song)
	return updated_songs_list

def song_to_vector(song, model):
    word_vectors = []
    for word in song:
        embeddings = model.wv[word]
        word_vectors.append(embeddings)

    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        # If no words in the song are in the vocabulary, return zeros
        return np.zeros(model.vector_size)

def build_models():
	print("build started ")
	train_happy = []
	folder_path = 'Data/Training_Data/Happy_Songs/'
	if os.path.isdir(folder_path):
    # Iterate over files in the directory
		for filename in os.listdir(folder_path):
			file_path = os.path.join(folder_path, filename)
			try:
				with open(file_path, 'r', encoding='utf-8') as f:
					train_happy.append(f.read())
			except UnicodeDecodeError:
				print(f"Error decoding file: {file_path}")



	train_sad = []
	folder_path = 'Data/Training_Data/Sad_Songs/'
	if os.path.isdir(folder_path):
    # Iterate over files in the directory
		for filename in os.listdir(folder_path):
			file_path = os.path.join(folder_path, filename)
			try:
				with open(file_path, 'r', encoding='utf-8') as f:
					train_sad.append(f.read())
			except UnicodeDecodeError:
				print(f"Error decoding file: {file_path}")

	train_happy = [list(song.split(' ')) for song in preprocess(train_happy)]
	train_sad = [list(song.split(' ')) for song in preprocess(train_sad)]

	pretrained_model_path = "indicnlp.ft.te.300.bin"
	model = FastText.load_fasttext_format(pretrained_model_path)
	training_data = train_happy + train_sad
	model.build_vocab(corpus_iterable=training_data , update=True)
	model.train(corpus_iterable=training_data , total_examples=len(training_data), epochs=10)
	fine_tuned_model_path = "fine_tuned_indicnlp.ft.te.300.bin"
	model.save(fine_tuned_model_path)

	train_happy_vectors = [song_to_vector(song, model) for song in train_happy]
	train_sad_vectors = [song_to_vector(song, model) for song in train_sad]

	training_Labels = [1 for i in train_happy] + [0 for i in train_sad]
	X_train = train_happy_vectors + train_sad_vectors
	y_train = training_Labels

	#Naive Bayes
	naive_bayes_model = GaussianNB()
	naive_bayes_model.fit(X_train, y_train)
	with open('naive_bayes_model.pkl', 'wb') as file:
		pickle.dump(naive_bayes_model, file)

	#Logistics Regression
	logistic_regression_model = LogisticRegression(max_iter=10000)
	logistic_regression_model.fit(X_train, y_train)
	with open('logistic_regression_model.pkl', 'wb') as file:
		pickle.dump(logistic_regression_model, file)

	#Support Vector Classifier
	support_vector_classifier_model = SVC(kernel='linear', random_state=33)
	support_vector_classifier_model.fit(X_train, y_train)
	with open('support_vector_classifier_model.pkl', 'wb') as file:
		pickle.dump(support_vector_classifier_model, file)

	#Decision Tree Classifier
	decision_tree_classifier_model = DecisionTreeClassifier()
	param_grid_decision_tree = {
	    'criterion': ['gini', 'entropy', 'log_loss'], 
	    'max_depth': [2,4, 5, 6, 7,10 , 15 , 20, None],
	    'min_samples_split': [2, 3, 5, 7, 9 , 12 , 15],
	    'min_samples_leaf': [1, 2, 4 , 6 , 8 , 10]
	}
	grid_search_decision_tree = GridSearchCV(estimator=decision_tree_classifier_model, param_grid=param_grid_decision_tree, cv=5, n_jobs=-1, verbose=1)
	grid_search_decision_tree.fit(X_train, y_train)
	best_decision_tree_classifier_model = grid_search_decision_tree.best_estimator_
	with open('best_decision_tree_classifier_model.pkl', 'wb') as file:
		pickle.dump(best_decision_tree_classifier_model, file)

	#Random Forest Classifier
	random_forest_classifier_model = RandomForestClassifier()
	param_grid_random_forest = {
	    'n_estimators': [20 , 30 , 40 ,50, 100, 200], # Number of trees in the forest
	    'max_depth': [3,4, 5,6, 7, 8 , 9,  None], 
	    'min_samples_split': [2, 3, 5 , 7 , 9 , 10 , 15],
	    'min_samples_leaf': [1, 2, 3, 4, 5 , 6 ],
	    'bootstrap': [True, False] # Whether to use bootstrap sampling
	}
	grid_search_random_forest = GridSearchCV(estimator=random_forest_classifier_model, param_grid=param_grid_random_forest, cv=2, n_jobs=-1, verbose=1)
	grid_search_random_forest.fit(X_train, y_train)
	best_random_forest_classifier_model = grid_search_random_forest.best_estimator_
	with open('best_random_forest_classifier_model.pkl', 'wb') as file:
		pickle.dump(best_random_forest_classifier_model, file)

	return "success"	


def predict_sentiment(song):

	songs = [list(song.split(' ')) for song in preprocess([song])]

	fine_tuned_model_path = "fine_tuned_indicnlp.ft.te.300.bin"
	indicft_model = FastText.load(fine_tuned_model_path)

	X_test = [song_to_vector(song, indicft_model) for song in songs]
	print(len(X_test))
	results_dictonary = {}

	with open('naive_bayes_model.pkl', 'rb') as file:
		naive_bayes_model = pickle.load(file)

	if(naive_bayes_model.predict(X_test)[0] == 1):
		results_dictonary["naive_bayes"] = "happy"
	else:
		results_dictonary["naive_bayes"] = "sad"


	with open('logistic_regression_model.pkl', 'rb') as file:
		logistic_regression_model = pickle.load(file)

	if(logistic_regression_model.predict(X_test)[0] == 1):
		results_dictonary["logistic_regression"] = "happy"
	else:
		results_dictonary["logistic_regression"] = "sad"


	with open('support_vector_classifier_model.pkl', 'rb') as file:
		support_vector_classifier_model = pickle.load(file)

	print(support_vector_classifier_model.predict(X_test)[0])
	if(support_vector_classifier_model.predict(X_test)[0] == 1):
		results_dictonary["support_vector_classifier"] = "happy"
	else:
		results_dictonary["support_vector_classifier"] = "sad"


	with open('best_decision_tree_classifier_model.pkl', 'rb') as file:
		best_decision_tree_classifier_model = pickle.load(file)

	if(best_decision_tree_classifier_model.predict(X_test)[0] == 1):
		results_dictonary["decision_tree_classifier"] = "happy"
	else:
		results_dictonary["decision_tree_classifier"] = "sad"


	with open('best_random_forest_classifier_model.pkl', 'rb') as file:
		best_random_forest_classifier_model = pickle.load(file)

	if(best_random_forest_classifier_model.predict(X_test)[0] == 1):
		results_dictonary["random_forest_classifier"] = "happy"
	else:
		results_dictonary["random_forest_classifier"] = "sad"


	return results_dictonary


# def main():
# 	# build_models()
# 	song = "ప్రాణం లో ప్రాణంగామాటల్లో మౌనంగా చెబుతున్నాబాధైనా ఏదైనాభారంగా దూరంగా వెళుతున్నామొన్న కన్నా కళా నిన్న విన్న కథారేపు రాదు కదా జతాఇలా ఇలా నిరాశగా దరిదాటుతున్నాఊరు మారుతున్నా ఊరుకోదు ఏదాప్రాణం లో ప్రాణంగామాటల్లో మౌనంగా చెబుతున్నాబాధైనా ఏదైనాభారంగా దూరంగా వెళుతున్నామొన్న కన్నా కళా నిన్న విన్న కథారేపు రాదు కదా జతాఇలా ఇలా నిరాశగా దరిదాటుతున్నాఊరు మారుతున్నా ఊరుకోదు ఏదాప్రాణం లో ప్రాణంగామాటల్లో మౌనంగా చెబుతున్నాస్నేహం నాదే ప్రేమా నాదేఆపైన ద్రోహం నాదేకన్నూ నాదే వేలు నాదేకన్నీరు నాదేలేతప్పంతా నాదే శిక్షన్త నాకేతప్పించుకోలేనేఎడారిలో తుఫానులా తడి ఆరుతున్నాతుది చూడకున్నా ఎదురీదుతున్నాప్రాణం లో ప్రాణంగామాటల్లో మౌనంగా చెబుతున్నాబాధైనా ఏదైనాభారంగా దూరంగా వెళుతున్నాఆటా నాదే గెలుపు నాదేఅనుకోని ఓటమి నాదేమాటా నాదే బదులూ నాదేప్రశ్నల్లె మిగిలానేనా జాతకాన్నీ నా చేతితోనేఏమార్చి రాసానేగతానిపై సమాధినై గతి మారుతున్నాస్థితి మారుతున్నా బ్రతికేస్తూ ఉన్నాప్రాణం లో ప్రాణంగామాటల్లో మౌనంగా చెబుతున్నాగతానిపై సమాధినై గతి మారుతున్నాస్థితి మారుతున్నా బ్రతికేస్తూ ఉన్నా"
# 	result = predict_sentiment(song)
# 	print(result)

# main()
	