import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

class TextProcessor:
    def __init__(self):
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize components
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        datasets = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
        
        for dataset in datasets:
            try:
                nltk.data.find(f'tokenizers/{dataset}')
            except LookupError:
                try:
                    nltk.download(dataset, quiet=True)
                except:
                    pass  # Ignore download errors
    
    def clean_text(self, text):
        """Basic text cleaning"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\-\']', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize(self, text):
        """Tokenize text into words"""
        try:
            tokens = word_tokenize(text)
        except:
            # Fallback tokenization
            tokens = text.split()
        
        return tokens
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from token list"""
        if not self.stop_words:
            return tokens
        
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize(self, tokens):
        """Lemmatize tokens"""
        try:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        except:
            return tokens  # Return original tokens if lemmatization fails
    
    def preprocess(self, text, remove_stopwords=False, lemmatize=False):
        """Complete text preprocessing pipeline"""
        if not text:
            return ""
        
        # Clean text
        cleaned = self.clean_text(text)
        
        # For basic preprocessing, just return cleaned text
        if not remove_stopwords and not lemmatize:
            return cleaned
        
        # Tokenize
        tokens = self.tokenize(cleaned)
        
        # Remove stopwords if requested
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Lemmatize if requested
        if lemmatize:
            tokens = self.lemmatize(tokens)
        
        return ' '.join(tokens)
    
    def extract_features(self, text):
        """Extract text features for analysis"""
        if not text:
            return {}
        
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        
        features = {
            'length': len(text),
            'word_count': len(tokens),
            'avg_word_length': sum(len(word) for word in tokens) / len(tokens) if tokens else 0,
            'punctuation_count': sum(1 for char in text if char in string.punctuation),
            'uppercase_count': sum(1 for char in text if char.isupper()),
            'digit_count': sum(1 for char in text if char.isdigit())
        }
        
        return features
