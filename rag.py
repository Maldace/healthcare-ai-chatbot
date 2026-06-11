from langchain_community.document_loaders import DirectoryLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = DirectoryLoader(
    path='./SympScan - Symptomps to Disease',
    glob='**/*.csv',
    loader_cls=CSVLoader,
    show_progress=True,
    use_multithreading=True
)

docs= loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    
)