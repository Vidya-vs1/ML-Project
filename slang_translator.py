import json
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import os
from utils.text_processor import TextProcessor

class SlangTranslator:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.dictionary_path = "data/slang_dictionary.json"
        self.training_data_path = "data/training_data.json"
        self.model_path = "models/slang_classifier.pkl"
        
        # Initialize components
        self.slang_dictionary = self.load_dictionary()
        self.training_data = self.load_training_data()
        self.classifier = None
        self.vectorizer = None
        self.similarity_vectors = None
        self.similarity_terms = None
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Train initial model
        self.train_classifier()
        self.prepare_similarity_matching()
    
    def load_dictionary(self):
        """Load the slang dictionary from file"""
        try:
            with open(self.dictionary_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def save_dictionary(self):
        """Save the slang dictionary to file"""
        os.makedirs(os.path.dirname(self.dictionary_path), exist_ok=True)
        with open(self.dictionary_path, 'w', encoding='utf-8') as f:
            json.dump(self.slang_dictionary, f, indent=2, ensure_ascii=False)
    
    def load_training_data(self):
        """Load training data from file"""
        try:
            with open(self.training_data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def save_training_data(self):
        """Save training data to file"""
        os.makedirs(os.path.dirname(self.training_data_path), exist_ok=True)
        with open(self.training_data_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_data, f, indent=2, ensure_ascii=False)
    
    def add_dictionary_entry(self, slang_term, standard_term):
        """Add a new entry to the slang dictionary"""
        slang_term = slang_term.lower().strip()
        standard_term = standard_term.strip()
        
        if slang_term in self.slang_dictionary:
            return False  # Entry already exists
        
        self.slang_dictionary[slang_term] = standard_term
        self.save_dictionary()
        
        # Update similarity matching
        self.prepare_similarity_matching()
        
        return True
    
    def add_training_data(self, slang_text, standard_text):
        """Add a new training pair"""
        training_pair = {
            "slang": slang_text.strip(),
            "standard": standard_text.strip()
        }
        
        self.training_data.append(training_pair)
        self.save_training_data()
        return True
    
    def train_classifier(self):
        """Train the Multinomial Naive Bayes classifier"""
        if len(self.training_data) < 2:
            # Not enough data to train
            return False
        
        # Prepare training data
        X_slang = [item['slang'] for item in self.training_data]
        y_standard = [item['standard'] for item in self.training_data]
        
        X_standard = [item['standard'] for item in self.training_data]
        y_slang = [item['slang'] for item in self.training_data]
        
        # Process text
        X_slang_processed = [self.text_processor.preprocess(text) for text in X_slang]
        X_standard_processed = [self.text_processor.preprocess(text) for text in X_standard]
        
        # Create pipelines for both directions
        self.slang_to_standard_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
            ('classifier', MultinomialNB(alpha=1.0))
        ])
        
        self.standard_to_slang_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
            ('classifier', MultinomialNB(alpha=1.0))
        ])
        
        try:
            # Train both pipelines
            self.slang_to_standard_pipeline.fit(X_slang_processed, y_standard)
            self.standard_to_slang_pipeline.fit(X_standard_processed, y_slang)
            
            # Save models
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'slang_to_standard': self.slang_to_standard_pipeline,
                    'standard_to_slang': self.standard_to_slang_pipeline
                }, f)
            
            return True
        except Exception as e:
            print(f"Error training classifier: {e}")
            return False
    
    def prepare_similarity_matching(self):
        """Prepare vectors for similarity matching"""
        # Combine dictionary and training data for similarity matching
        all_terms = list(self.slang_dictionary.keys())
        all_meanings = list(self.slang_dictionary.values())
        
        for item in self.training_data:
            all_terms.append(item['slang'])
            all_meanings.append(item['standard'])
        
        if not all_terms:
            return
        
        # Create TF-IDF vectors for similarity matching
        self.similarity_vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        
        processed_terms = [self.text_processor.preprocess(term) for term in all_terms]
        self.similarity_vectors = self.similarity_vectorizer.fit_transform(processed_terms)
        self.similarity_terms = list(zip(all_terms, all_meanings))
    
    def dictionary_lookup(self, text, slang_to_standard=True):
        """Perform dictionary-based lookup"""
        text_lower = text.lower().strip()
        
        if slang_to_standard:
            # Look up slang term in dictionary
            if text_lower in self.slang_dictionary:
                return {
                    'translation': self.slang_dictionary[text_lower],
                    'confidence': 1.0,
                    'method': 'Dictionary',
                    'alternatives': []
                }
        else:
            # Reverse lookup: find slang for standard English
            for slang, standard in self.slang_dictionary.items():
                if standard.lower() == text_lower:
                    return {
                        'translation': slang,
                        'confidence': 1.0,
                        'method': 'Dictionary',
                        'alternatives': []
                    }
        
        return None
    
    def ml_prediction(self, text, slang_to_standard=True, confidence_threshold=0.5):
        """Use ML classifier for prediction"""
        if not hasattr(self, 'slang_to_standard_pipeline') or not hasattr(self, 'standard_to_slang_pipeline'):
            return None
        
        try:
            processed_text = self.text_processor.preprocess(text)
            
            if slang_to_standard:
                pipeline = self.slang_to_standard_pipeline
            else:
                pipeline = self.standard_to_slang_pipeline
            
            # Get prediction and probability
            prediction = pipeline.predict([processed_text])[0]
            probabilities = pipeline.predict_proba([processed_text])[0]
            
            confidence = max(probabilities)
            
            if confidence >= confidence_threshold:
                return {
                    'translation': prediction,
                    'confidence': confidence,
                    'method': 'ML Classifier',
                    'alternatives': []
                }
        except Exception as e:
            print(f"ML prediction error: {e}")
        
        return None
    
    def similarity_matching(self, text, slang_to_standard=True, top_k=3):
        """Use cosine similarity for finding closest matches"""
        if not hasattr(self, 'similarity_vectorizer') or self.similarity_vectors is None:
            return None
        
        try:
            processed_text = self.text_processor.preprocess(text)
            text_vector = self.similarity_vectorizer.transform([processed_text])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(text_vector, self.similarity_vectors)[0]
            
            # Get top matches
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            best_match_idx = top_indices[0]
            best_similarity = similarities[best_match_idx]
            
            if best_similarity > 0.1:  # Minimum similarity threshold
                source_term, target_term = self.similarity_terms[best_match_idx]
                
                # Choose correct direction
                if slang_to_standard:
                    translation = target_term
                else:
                    translation = source_term
                
                # Get alternatives
                alternatives = []
                for idx in top_indices[1:]:
                    if similarities[idx] > 0.05:
                        alt_source, alt_target = self.similarity_terms[idx]
                        alt_translation = alt_target if slang_to_standard else alt_source
                        alternatives.append(f"{alt_translation} (similarity: {similarities[idx]:.2f})")
                
                return {
                    'translation': translation,
                    'confidence': best_similarity,
                    'method': 'Similarity',
                    'alternatives': alternatives
                }
        except Exception as e:
            print(f"Similarity matching error: {e}")
        
        return None
    
    def translate(self, text, slang_to_standard=True, confidence_threshold=0.5):
        """Main translation function using hybrid approach"""
        if not text or not text.strip():
            return {
                'translation': '',
                'confidence': 0.0,
                'method': 'None',
                'alternatives': []
            }
        
        # Method 1: Dictionary lookup
        result = self.dictionary_lookup(text, slang_to_standard)
        if result:
            return result
        
        # Method 2: ML classifier
        result = self.ml_prediction(text, slang_to_standard, confidence_threshold)
        if result:
            return result
        
        # Method 3: Similarity matching
        result = self.similarity_matching(text, slang_to_standard)
        if result:
            return result
        
        # No translation found
        return {
            'translation': f"Translation not found for: '{text}'",
            'confidence': 0.0,
            'method': 'None',
            'alternatives': ['Consider adding this term to the dictionary or training data']
        }
    
    def retrain_model(self):
        """Retrain the ML classifier with current data"""
        return self.train_classifier()
    
    def get_model_stats(self):
        """Get statistics about the model"""
        accuracy = 0.0
        
        # Calculate cross-validation accuracy if we have enough data
        if len(self.training_data) >= 5:
            try:
                X = [self.text_processor.preprocess(item['slang']) for item in self.training_data]
                y = [item['standard'] for item in self.training_data]
                
                pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
                    ('classifier', MultinomialNB(alpha=1.0))
                ])
                
                scores = cross_val_score(pipeline, X, y, cv=min(5, len(self.training_data)), scoring='accuracy')
                accuracy = scores.mean()
            except:
                accuracy = 0.0
        
        return {
            'dictionary_size': len(self.slang_dictionary),
            'training_size': len(self.training_data),
            'accuracy': accuracy
        }
    
    def get_dictionary(self):
        """Get the current dictionary"""
        return self.slang_dictionary.copy()
    
    def get_training_data(self):
        """Get the current training data"""
        return self.training_data.copy()
