import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

# Haber sitesinden haber metnini çekmek için bir fonksiyon
def get_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p') # Paragrafları al
    text = ' '.join([p.get_text() for p in paragraphs]) # Paragrafları birleştir
    return text

# Metin ön işleme: Noktalama işaretlerini, durma kelimelerini ve fazlalıkları kaldırma
def preprocess_text(text):
    sentences = sent_tokenize(text)
    clean_sentences = [s.lower() for s in sentences]
    clean_sentences = [s for s in clean_sentences if len(s) > 0]
    return clean_sentences

# Özetleme fonksiyonu
def summarize_text(text, num_sentences=3):
    sentences = preprocess_text(text)

    # TF-IDF matrisini oluşturma
    stop_words = stopwords.words('turkish')
    word_frequencies = {}
    for sentence in sentences:
        for word in sentence.split():
            if word not in stop_words:
                if word not in word_frequencies:
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency

    # Cümle skorlarını hesaplama
    sentence_scores = {}
    for sentence in sentences:
        for word in sentence.split():
            if word in word_frequencies:
                if len(sentence.split()) < 30:
                    if sentence not in sentence_scores:
                        sentence_scores[sentence] = word_frequencies[word]
                    else:
                        sentence_scores[sentence] += word_frequencies[word]

    # En önemli cümlelerin seçilmesi
    ranked_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    selected_sentences = [s[0] for s in ranked_sentences[:num_sentences]]

    return ' '.join(selected_sentences)

# Ana işlem
def main():
    # Özetlemek istediğiniz haberin URL'sini buraya girin
    url = 'https://www.milliyet.com.tr/gundem/son-dakika-cumhurbaskani-erdogandan-sanliurfada-onemli-aciklamalar-7091683'
    news_text = get_text_from_url(url)
    summary = summarize_text(news_text)
    print("Haber Özeti:")
    print(summary)

if __name__ == "__main__":
    main()
