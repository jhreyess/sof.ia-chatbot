from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
from faq_data import faq_dict
import joblib

# Uncomment this lines for the first running time
# nltk.download('stopwords')
# nltk.download('punkt')

class SofiaAgent:
    def __init__(self, model_path, vectorizer_path):

        self.stop_words = set(stopwords.words('spanish'))
        self.stemmer = SnowballStemmer('spanish')
        self.db = faq_dict
        
        # Load the vectorizer & Load the decision from file
        if model_path is not None:
            self.model = joblib.load(model_path)
        else:
            self.model = None

        if vectorizer_path is not None:
            self.vectorizer = joblib.load(vectorizer_path)
        else:
            self.vectorizer = None

    def ask(self, perception, major):
        features = self.extract_features(perception)
        label = self.model.predict(features)[0]

        # Optional
        predicted_prob = self.model.predict_proba(features)[0]

        answer = self.answer_question(perception, label, major)
        return answer, label, "preprocessed_input", predicted_prob
    
    def clean_question_text(self, question):
        print("Question is: ", question)
        # Remove question marks and other punctuation marks
        question = question.replace("¿", "").replace("?", "")
        # Split the question into words (Tokenization)
        words = question.lower().split()
        # Remove stop words and stem the words
        words = [self.stemmer.stem(w) for w in words if w not in self.stop_words]
        # Join the words back into a string and return it
        return " ".join(words)

    def extract_features_tokenizer(self, question):
        print("Messsage is: ", question)
        words = question.split()
        # Generate bigrams
        bigrams = [" ".join(words[i:i+2]) for i in range(len(words) - 1)]
        # Combine unigrams and bigrams
        tokens = words + bigrams
        print(f"{question} => {tokens}")
        return tokens

    def extract_features(self, perception):
        # Preprocess the question
        preprocessed_input = self.clean_question_text(perception)
        # Preprocess and transform the preprocessed question using the vectorizer
        vectorized_input = self.vectorizer.transform([preprocessed_input])
        print(vectorized_input)
        return vectorized_input

    def answer_question(self, perception, label, major):
        category = self.db.get(label)

        if(label == "majors"):
            # Combine major-specific and general questions into a single dictionary
            major_category = category.get(major, {})
            general_category = category.get("general", {})
            category = { **general_category, **major_category }

        highest_ratio = 0
        matching_question = None
        for question in category.keys():
            ratio = fuzz.token_sort_ratio(perception.lower(), question.lower())
            if ratio > highest_ratio:
                highest_ratio = ratio
                matching_question = question
                print("Ratio: ", highest_ratio)
        
        if(highest_ratio < 50):
            return f"Lo siento, no pude encontrar información relacionada, pero entiendo que buscas información sobre {label}, prueba reformulando tu pregunta."
        return category[matching_question]