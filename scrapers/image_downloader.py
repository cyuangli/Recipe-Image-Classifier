import os
import ast
import requests
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import pandas as pd
import re
import time
import random
import math


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATAFRAME_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "..", "notebooks", "data", "recipe_meta_topics.csv")
)

IMAGES_DIR = os.path.normpath(
    os.path.join(BASE_DIR, "..", "notebooks", "data", "images")
)

IMAGE_SIZE = (224, 224)
JPEG_QUALITY = 85
MAX_RETRIES = 3

START_INDEX = 0

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:117.0) Gecko/20100101 Firefox/117.0",
]



def sanitize_filename(name, fallback):
    if not isinstance(name, str) or not name.strip():
        name = fallback
    name = name.lower()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name

def download_image(url, save_path):
    for attempt in range(1, MAX_RETRIES + 1):
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert("RGB")
                img = img.resize(IMAGE_SIZE)
                img.save(save_path, format="JPEG", quality=JPEG_QUALITY)
                return True
            else:
                print(f"Attempt {attempt}: Failed {url} (status {response.status_code})")
        except Exception as e:
            print(f"Attempt {attempt}: Error downloading {url}: {e}")

        time.sleep(random.uniform(0.5, 2.0))
    return False


df = pd.read_csv(DATAFRAME_PATH)
os.makedirs(IMAGES_DIR, exist_ok=True)

total_recipes = len(df)


for idx, row in tqdm(df.iterrows(), total=total_recipes):
    if idx < START_INDEX:
        continue

    recipe_name = row.get("name")
    meta_topic = row.get("meta_topic")

    if pd.isna(meta_topic):
        continue

    filename_base = sanitize_filename(
        recipe_name,
        fallback=f"recipe_{idx}"
    )

    meta_dir = os.path.join(IMAGES_DIR, f"meta_topic_{int(meta_topic)}")
    os.makedirs(meta_dir, exist_ok=True)

    save_path = os.path.join(meta_dir, f"{filename_base}_{idx}.jpg")

    if os.path.exists(save_path):
        continue

    try:
        image_list = ast.literal_eval(row["images"])
    except Exception:
        continue

    if not image_list:
        continue

    url = image_list[0]

    success = download_image(url, save_path)
    if not success:
        print(f"Failed after {MAX_RETRIES} attempts: {url}")

    time.sleep(random.uniform(0.3, 1.0))
