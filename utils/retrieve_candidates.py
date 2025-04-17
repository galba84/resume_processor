def retrieve_candidate_details(indices, resumes):
    candidates = []
    for index in indices:
        candidates.append(resumes[index])
    return candidates
