def summarize_experience(resume):
    # Simple example, you can use more advanced NLP techniques
    lines = resume.split('\n')
    summary = " ".join(lines[:5])  # First 5 lines as summary
    return summary
