from openai import OpenAI
import os
import requests
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"


# def download_file(url, local_filename):
#     with requests.get(url, stream=True) as r:
#         r.raise_for_status()
#         with open(local_filename, 'wb') as file:
#             for chunk in r.iter_content(chunk_size=8192):
#                 file.write(chunk)
#     return local_filename

def download_file(url, local_filename):
    session = requests.Session()
    retries = Retry(total=100, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))

    with session.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as file:
            for chunk in r.iter_content(chunk_size=8192):
                file.write(chunk)
    return local_filename


def generate_by_dall_e3(text):
    client = OpenAI(api_key="api_key")

    response = client.images.generate(
        model="dall-e-3",
        prompt=text,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url
    return image_url

def generate_picture(directory, name, num):
    with open(os.path.join(directory, f'{name}.txt'), 'r', encoding='utf-8') as file:
        origin_text = "Generate a paper illustration based on the following content:\n" + f"\"\"\"\n{file.read()}\n\"\"\""
        text = origin_text[:1000]
        print(text)
        print(len(text))
        start_time = time.time()
        image_url = generate_by_dall_e3(text)
        cost = time.time() - start_time
        print(cost)
        print(image_url)
        local_filename = os.path.join(directory, f"{name}.png")
        with open(os.path.join(directory, f"image_time.csv"), 'a') as outfile:
            outfile.write(f"image{num}, {cost:.2f}\n")
        download_file(image_url, local_filename)
# if __name__ == "__main__":
#     labels = ['Introduction', 'Related work', 'Methodology', 'Results', 'Future work', 'Conclusion']
#     t = [2, 3]
#     for word_count in range(6000, 7000, 1000):
#         for idx in range(9, 11, 1):
#             directory = f"./word_{word_count}_{idx}"
#             # if os.path.exists(os.path.join(directory, f"image_time.csv")):
#             #     os.remove(os.path.join(directory, f"image_time.csv"))
#             for num in t:
#                 with open(os.path.join(directory, f"{labels[num]}.txt"), 'r', encoding='utf-8') as file:
#                     origin_text = "Generate a paper illustration based on the following content:\n" + f"\"\"\"\n{file.read()}\n\"\"\""
#                     text = origin_text[:1000]
#                     print(text)
#                     print(len(text))
#                     start_time = time.time()
#                     image_url = generate_by_dall_e3(text)
#                     cost = time.time() - start_time
#                     print(cost)
#                     print(image_url)
#                     local_filename = os.path.join(directory, f"{labels[num]}.png")
#                     with open(os.path.join(directory, f"image_time.csv"), 'a') as outfile:
#                         outfile.write(f"image{num}, {cost:.2f}\n")
#                     download_file(image_url, local_filename)
#             print(word_count, idx)
