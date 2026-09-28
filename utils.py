def normalize_score(score, maximum):
    if maximum <= 0:
        raise ValueError("Maximum must be greater than zero")
    return score / maximum