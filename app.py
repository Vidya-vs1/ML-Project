import streamlit as st
import json
import pandas as pd
import numpy as np
from slang_translator import SlangTranslator
import os

# Configure page
st.set_page_config(
    page_title="Gen Z Slang Translator",
    page_icon="🔤",
    layout="wide"
)

# Initialize translator
@st.cache_resource
def load_translator():
    return SlangTranslator()

def main():
    st.title("🔤 Gen Z Slang Translator")
    st.markdown("**Bridge the generational communication gap with AI-powered slang translation**")
    
    # Load translator
    translator = load_translator()
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Translation direction
    direction = st.sidebar.selectbox(
        "Translation Direction",
        ["Gen Z Slang → Standard English", "Standard English → Gen Z Slang"],
        help="Choose the direction of translation"
    )
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Minimum confidence score for ML predictions"
    )
    
    # Show method details
    show_details = st.sidebar.checkbox(
        "Show Translation Method Details",
        value=True,
        help="Display which method was used for each translation"
    )
    
    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔄 Translate", "📊 Model Info", "📚 Dictionary", "🎯 Training"])
    
    with tab1:
        translate_tab(translator, direction, confidence_threshold, show_details)
    
    with tab2:
        model_info_tab(translator)
    
    with tab3:
        dictionary_tab(translator)
    
    with tab4:
        training_tab(translator)

def translate_tab(translator, direction, confidence_threshold, show_details):
    st.header("Translation Interface")
    
    # Input method selection
    input_method = st.radio(
        "Input Method",
        ["Single Text", "Batch Translation"],
        horizontal=True
    )
    
    if input_method == "Single Text":
        single_translation(translator, direction, confidence_threshold, show_details)
    else:
        batch_translation(translator, direction, confidence_threshold, show_details)

def single_translation(translator, direction, confidence_threshold, show_details):
    # Text input
    input_text = st.text_area(
        "Enter text to translate:",
        height=100,
        placeholder="Type your text here..."
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        translate_button = st.button("🔄 Translate", type="primary")
    
    if translate_button and input_text.strip():
        with st.spinner("Translating..."):
            is_slang_to_standard = direction == "Gen Z Slang → Standard English"
            result = translator.translate(
                input_text, 
                slang_to_standard=is_slang_to_standard,
                confidence_threshold=confidence_threshold
            )
            
            display_translation_result(result, show_details)

def batch_translation(translator, direction, confidence_threshold, show_details):
    st.markdown("**Enter multiple phrases (one per line):**")
    
    batch_input = st.text_area(
        "Batch Input:",
        height=150,
        placeholder="Line 1: First phrase\nLine 2: Second phrase\nLine 3: Third phrase..."
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        translate_button = st.button("🔄 Translate All", type="primary")
    
    if translate_button and batch_input.strip():
        lines = [line.strip() for line in batch_input.split('\n') if line.strip()]
        
        if lines:
            with st.spinner(f"Translating {len(lines)} phrases..."):
                results = []
                is_slang_to_standard = direction == "Gen Z Slang → Standard English"
                
                for i, line in enumerate(lines):
                    result = translator.translate(
                        line,
                        slang_to_standard=is_slang_to_standard,
                        confidence_threshold=confidence_threshold
                    )
                    results.append({
                        'Input': line,
                        'Translation': result['translation'],
                        'Confidence': result['confidence'],
                        'Method': result['method'],
                        'Alternatives': ', '.join(result.get('alternatives', []))
                    })
                
                # Display results in a table
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                
                # Download option
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="translation_results.csv",
                    mime="text/csv"
                )

def display_translation_result(result, show_details):
    # Main translation result
    st.success("**Translation Result:**")
    st.markdown(f"### {result['translation']}")
    
    # Confidence and method info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        confidence_color = "green" if result['confidence'] >= 0.7 else "orange" if result['confidence'] >= 0.4 else "red"
        st.metric(
            "Confidence",
            f"{result['confidence']:.2%}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Method Used",
            result['method'],
            delta=None
        )
    
    with col3:
        if result.get('alternatives'):
            st.metric(
                "Alternatives Found",
                len(result['alternatives']),
                delta=None
            )
    
    # Method details
    if show_details:
        with st.expander("🔍 Translation Method Details"):
            method = result['method']
            if method == "Dictionary":
                st.info("✅ **Direct Dictionary Match**: This translation was found in our curated slang dictionary.")
            elif method == "ML Classifier":
                st.info("🤖 **Machine Learning Prediction**: Our trained Naive Bayes classifier predicted this translation using TF-IDF features.")
            elif method == "Similarity":
                st.info("🔍 **Similarity Matching**: Used cosine similarity to find the closest match from known translations.")
            
            if result.get('alternatives'):
                st.markdown("**Alternative Suggestions:**")
                for alt in result['alternatives']:
                    st.markdown(f"• {alt}")

def model_info_tab(translator):
    st.header("📊 Model Performance & Statistics")
    
    # Model statistics
    stats = translator.get_model_stats()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Dictionary Entries",
            stats['dictionary_size'],
            help="Number of direct slang-to-standard mappings"
        )
    
    with col2:
        st.metric(
            "Training Samples",
            stats['training_size'],
            help="Number of samples used to train the ML classifier"
        )
    
    with col3:
        st.metric(
            "Model Accuracy",
            f"{stats.get('accuracy', 0):.2%}",
            help="Cross-validation accuracy of the ML classifier"
        )
    
    # Feature information
    st.subheader("🔧 Technical Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Dictionary-based Translation:**")
        st.markdown("- Primary method for known slang terms")
        st.markdown("- Instant, high-confidence results")
        st.markdown("- Manually curated entries")
        
        st.markdown("**Machine Learning Classifier:**")
        st.markdown("- Multinomial Naive Bayes algorithm")
        st.markdown("- TF-IDF vectorization for features")
        st.markdown("- Trained on slang-standard pairs")
    
    with col2:
        st.markdown("**Similarity Matching:**")
        st.markdown("- Cosine similarity calculation")
        st.markdown("- Fallback for unknown terms")
        st.markdown("- Suggests closest matches")
        
        st.markdown("**Confidence Scoring:**")
        st.markdown("- Dictionary matches: 100% confidence")
        st.markdown("- ML predictions: Based on probability")
        st.markdown("- Similarity matches: Based on cosine score")

def dictionary_tab(translator):
    st.header("📚 Slang Dictionary")
    
    # Display current dictionary
    dictionary = translator.get_dictionary()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Current Dictionary Entries")
        
        # Search functionality
        search_term = st.text_input("🔍 Search dictionary:", placeholder="Enter slang term...")
        
        # Filter dictionary based on search
        if search_term:
            filtered_dict = {k: v for k, v in dictionary.items() 
                           if search_term.lower() in k.lower() or search_term.lower() in v.lower()}
        else:
            filtered_dict = dictionary
        
        # Display dictionary as DataFrame
        if filtered_dict:
            df = pd.DataFrame(list(filtered_dict.items()), columns=['Slang Term', 'Standard English'])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No matching entries found.")
    
    with col2:
        st.subheader("Add New Entry")
        
        new_slang = st.text_input("Slang term:")
        new_standard = st.text_input("Standard English:")
        
        if st.button("➕ Add Entry") and new_slang and new_standard:
            success = translator.add_dictionary_entry(new_slang.lower(), new_standard)
            if success:
                st.success(f"Added: {new_slang} → {new_standard}")
                st.rerun()
            else:
                st.warning("Entry already exists!")

def training_tab(translator):
    st.header("🎯 Model Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Add Training Data")
        
        train_slang = st.text_area("Slang text:", height=100)
        train_standard = st.text_area("Standard English:", height=100)
        
        if st.button("➕ Add Training Pair") and train_slang and train_standard:
            success = translator.add_training_data(train_slang, train_standard)
            if success:
                st.success("Training pair added successfully!")
            else:
                st.warning("Failed to add training pair.")
    
    with col2:
        st.subheader("Retrain Model")
        
        st.markdown("Current model performance:")
        stats = translator.get_model_stats()
        st.metric("Training Samples", stats['training_size'])
        
        if st.button("🔄 Retrain Classifier"):
            with st.spinner("Retraining model..."):
                success = translator.retrain_model()
                if success:
                    st.success("Model retrained successfully!")
                    st.rerun()
                else:
                    st.error("Failed to retrain model.")
    
    # Display current training data
    st.subheader("Current Training Data")
    training_data = translator.get_training_data()
    
    if training_data:
        df = pd.DataFrame(training_data)
        st.dataframe(df, use_container_width=True)
        
        # Download training data
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Training Data",
            data=csv,
            file_name="training_data.csv",
            mime="text/csv"
        )
    else:
        st.info("No training data available.")

if __name__ == "__main__":
    main()
