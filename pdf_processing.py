from PyPDF2 import PdfReader


def extract_text_from_pdf(filepath):
    with open(filepath, "rb") as file:
        reader = PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + " "
    text = text.replace('\n', ' ').replace('"', '').replace("'", '')
    return text

