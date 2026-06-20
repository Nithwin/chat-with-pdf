from pypdf import PdfReader

reader = PdfReader("resume.pdf")

text = ""

for page in reader.pages:
    extracted_text = page.extract_text()

    if extracted_text:
        text += extracted_text + "\n"
    

chunks = [
    text[i: i+3]
    for i in range(0, len(text), 3)
]

