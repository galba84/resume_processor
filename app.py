import PyPDF2
import os
import glob
from transformers import pipeline
from flask import Flask, render_template, url_for

app = Flask(__name__)


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


@app.route('/')
def index():
    # Process all PDFs in the specified directory

    directory = r'C:\Users\olsereda\.cache\kagglehub\datasets\snehaanbhawal\resume-dataset\versions\1\data\data\ACCOUNTANT'
    all_texts = process_all_pdfs(directory)
    candidates = [f'Candidate {i + 1}' for i in range(len(all_texts))]
    return render_template('index.html', candidates=candidates)


@app.route('/candidate/<id>')
def candidate(id):
    # Retrieve candidate details based on the ID

    directory = r'C:\Users\olsereda\.cache\kagglehub\datasets\snehaanbhawal\resume-dataset\versions\1\data\data\ACCOUNTANT'
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
    return render_template('candidate_details.html', resume=candidate_details['resume'], summary=summary)


if __name__ == '__main__':
    app.run(debug=True)
