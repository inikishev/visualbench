import shutil
import zipfile


def load_movie_lens():
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    output_filename = "ml-100k.zip"

    import requests
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(output_filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    with zipfile.ZipFile("ml-100k.zip","r") as zip_ref:
        zip_ref.extractall("MovieLens-100k")

    return "MovieLens-100k/ml-100k"

def colab_pack():
    from google.colab import files # pylint:disable=import-error,no-name-in-module

    shutil.make_archive("MLBench", 'zip', "MLBench")

    # Download the file
    files.download('MLBench.zip')
