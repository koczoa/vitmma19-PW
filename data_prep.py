import requests
from tqdm import tqdm
import zipfile
import os
import shutil
import json
from enum import Enum
from collections import Counter
import glob
import pandas as pd

dataset_link = "https://bmeedu-my.sharepoint.com/:u:/g/personal/gyires-toth_balint_vik_bme_hu/IQB8kDcLEuTqQphHx7pv4Cw5AW7XMJp5MUbwortTASU223A?e=Uu6CTj&download=1"

if not os.path.exists("data.zip"):
    response = requests.get(dataset_link, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open("data.zip", "wb") as file, tqdm(desc="data.zip", total=total_size, unit='iB', unit_scale=True, unit_divisor=1024,) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            size = file.write(chunk)
            bar.update(size)

    with zipfile.ZipFile("data.zip", "r") as zip_ref:
        zip_ref.extractall(".")
    shutil.rmtree("anklealign/consensus")
    shutil.rmtree("anklealign/sample")
    os.remove("anklealign/ODZF0M/exploration.ipynb")
    for f in glob.glob("anklealign/FO6K58/ITWQ3V*.jpg"):
        os.remove(f)

    shutil.copytree("anklealign/NC1O2T/normal", "anklealign/NC1O2T/", dirs_exist_ok=True)
    shutil.copytree("anklealign/NC1O2T/pronation", "anklealign/NC1O2T/", dirs_exist_ok=True)
    shutil.copytree("anklealign/NC1O2T/supination", "anklealign/NC1O2T/", dirs_exist_ok=True)
    shutil.rmtree("anklealign/NC1O2T/normal")
    shutil.rmtree("anklealign/NC1O2T/pronation")
    shutil.rmtree("anklealign/NC1O2T/supination")
    shutil.rmtree("anklealign/GI9Y8B/")




labels = {}

class Annotation(Enum):
    PRO = "1_Pronacio"
    NEU = "2_Neutralis"
    SUP = "3_Szupinacio"
    NON = "bad"

os.chdir("anklealign")
for neptun in os.listdir():
    os.chdir(neptun)
    for file in os.listdir():
        if file.endswith(".json"):
            jsonfile = json.load(open(file))
            # annotation = [r['value']['choices'][0] for d in jsonfile for a in d['annotations'] for r in a['result'] if 'choices' in r.get('value', {})]
            for item in jsonfile:
                annot = Annotation.NON
                try:
                    annot = Annotation([result["value"]["choices"][0] for annotation in item["annotations"] for result in annotation["result"]][0])
                except ValueError:
                    match [result["value"]["choices"][0] for annotation in item["annotations"] for result in annotation["result"]][0]:
                        case "neutral":
                            annot = Annotation.NEU
                        case "supination":
                            annot = Annotation.SUP
                        case "pronation":
                            annot = Annotation.PRO
                except IndexError:
                    annot = Annotation.NON
                filename = f"anklealign/{neptun}/{item["file_upload"][9:]}"
                # print(filename, annot.value)
                labels[filename] = annot.value


    os.chdir("..")
os.chdir("..")

print("-------------------------- INITIAL DATA DISTRIBUTION --------------------------")
print(Counter(labels.values()))
print(len(labels))
print("-------------------------- INITIAL DATA DISTRIBUTION --------------------------")


df = pd.DataFrame(list(labels.items()), columns=["files", "annotations"])
df.drop(df[df["annotations"] == "bad"].index, inplace=True)
df.to_csv("labels.csv", index=False)
print(len(df))