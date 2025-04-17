import PyPDF2
import os
import glob
from transformers import pipeline
from flask import Flask, render_template, url_for
import chromadb
import numpy as np

app = Flask(__name__)


class EmbeddingsStore:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name="resume_embeddings")

    def add_embeddings(self, embeddings, ids, documents):
        self.collection.add(embeddings=embeddings, ids=ids, documents=documents)

    def search(self, query_embedding, k=5):
        results = self.collection.query(query_embeddings=[query_embedding], n_results=k)
        return results['ids'][0]


embeddings_store = EmbeddingsStore()


def load_resumes(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ''


def split_into_chunks(text, chunk_size=1000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def process_all_pdfs(directory):
    all_texts = []
    for file_path in glob.glob(os.path.join(directory, '**', '*.pdf'), recursive=True):
        text = load_resumes(file_path)
        if text:
            all_texts.append(text)
    return all_texts


def generate_summary(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    chunks = split_into_chunks(text)
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return ' '.join(summaries)


def generate_embeddings(text):
    embedder = pipeline("feature-extraction", model="distilbert-base-uncased")
    embeddings = embedder(text)
    # Ensure the embeddings are of the correct size
    if len(embeddings[0]) != 512:
        embeddings = np.resize(embeddings[0], (512,))
    else:
        embeddings = np.mean(embeddings[0], axis=0)
    return embeddings



@app.route('/')
def index():
    directory = 'C:/Users/olsereda/.cache/kagglehub/datasets/snehaanbhawal/resume-dataset/versions/1/data/data/ACCOUNTANT'
    all_texts = process_all_pdfs(directory)
    candidates = [f'Candidate {i + 1}' for i in range(len(all_texts))]

    # Add embeddings to the store
    for i, text in enumerate(all_texts):
        embedding = generate_embeddings(text)
        embeddings_store.add_embeddings([embedding], [f'Candidate {i + 1}'], [text])

    return render_template('index.html', candidates=candidates)


@app.route('/candidate/<id>')
def candidate(id):
    directory = 'C:/Users/olsereda/.cache/kagglehub/datasets/snehaanbhawal/resume-dataset/versions/1/data/data/ACCOUNTANT'
    all_texts = process_all_pdfs(directory)
    candidate_index = int(id.split(' ')[1]) - 1
    resume_text = all_texts[candidate_index]

    candidate_details = {
        'name': f'Candidate {candidate_index + 1}',
        'profession': 'Staff Accountant',
        'years_of_experience': '10+ years',
        'resume': resume_text
    }
    summary = generate_summary(candidate_details['resume'])
    return render_template('candidate.html', resume=candidate_details['resume'], summary=summary)


@app.route('/search', methods=['POST'])
def search():
    query_text = request.form['query']
    query_embedding = generate_embeddings(query_text)
    candidate_id = embeddings_store.search(query_embedding)
    return redirect(url_for('candidate', id=candidate_id))


if __name__ == '__main__':
    app.run(debug=True)
