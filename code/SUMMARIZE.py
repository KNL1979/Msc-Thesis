import pandas as pd
import deepl
from transformers import BartTokenizer, BartForConditionalGeneration
from tqdm import tqdm

def run_summarization(params):
    input_file = params.get('input_file')
    output_file = params.get('output_file')
    target_language = params.get('target_language', 'EN-US')
    api_key = params.get('API')
    num_beams = params.get('num_beams', 2)
    max_length_toke = params.get('max_length_tokenizer', 1024)
    max_length = params.get('max_length_summary', 150)
    min_length = params.get('min_length_summary', 80) 

    # Initialize the DeepL Translator
    translator = deepl.Translator(api_key)

    # Initialize the BART summarizer
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    # Load data
    df = pd.read_csv(input_file)

    def translate_text(text, target_lang):
        try:
            result = translator.translate_text(text, target_lang=target_lang)
            return result.text
        except Exception as e:
            print(f"An error occurred during translation: {e}")
            return text

    def summarize_text(text):
        inputs = tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=max_length_toke, truncation=True)
        summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=5.0, num_beams=num_beams)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Apply translation
    tqdm.pandas(desc="Translating")
    df['translated_text'] = df['text'].progress_apply(lambda x: translate_text(x, target_language))

    # Apply summarization
    tqdm.pandas(desc="Summarizing")
    df['summary'] = df['translated_text'].progress_apply(summarize_text)

    # Save the DataFrame with the translated text and summaries
    df.to_csv(output_file, index=False)

