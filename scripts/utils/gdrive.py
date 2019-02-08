from urllib import urlopen

import requests, json
from tqdm import tqdm

def download_file_from_google_drive(id, destination=None):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    tokens = response.headers['Content-Disposition'].split(';')
    filename_token = filter(lambda w: w.startswith('filename='), tokens)
    token = filename_token[0]
    filename = token[token.index('=')+2:-1]
    filepath = destination + '/' + filename
    save_response_content(response, filepath)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == '__main__':
    file_id = "1fmwEEdL0F-zxeKSx8ShYYl_KUPB2ilpE"
    download_file_from_google_drive(file_id,)