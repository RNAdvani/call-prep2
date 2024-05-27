import os
from fastapi import UploadFile


def save_file(file:UploadFile,upload_folder:str) -> str:
    os.makedirs(upload_folder,exist_ok=True)
    filepath = os.path.join(upload_folder,file.filename)
    with open(filepath) as f:
        f.write(file.file.read())
    return filepath
