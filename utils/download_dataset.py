import kagglehub

def download_dataset():
    path = kagglehub.dataset_download("snehaanbhawal/resume-dataset")
    print("Path to dataset files:", path)
    return path
