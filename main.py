with open("sample_document.txt", 'r') as file:
    document = file.read()
    # print(document)

    chunks = document.split("\n\n")
    # print(chunks)
    questions = input("User: ")
    for chunk in chunks:
        if questions.lower() in chunk.lower():
            print(chunk)