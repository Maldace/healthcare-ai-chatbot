# %%
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sklearn.metrics.pairwise import cosine_similarity
import json

# %%
df = pd.read_csv('./SympScan - Symptomps to Disease/Diseases_and_Symptoms_dataset.csv', encoding='utf-8')

docs = []
ten_benh = df['bệnh'].unique()
for benh in ten_benh:
    trieu_chung = []
    cac_mau_benh = df[df['bệnh']==benh]
    for i, row in cac_mau_benh.iterrows():
        trieu_chung.extend(col for col in cac_mau_benh.columns if col !='bệnh' and col not in trieu_chung and row[col]==1)
    cac_trieu_chung = ', '.join(trieu_chung)
    chi_tiet_benh = f'{benh} có các triệu chứng như: {cac_trieu_chung}.'
    doc=Document(page_content=chi_tiet_benh,metadata={"source":"Diseases_and_Symptoms_dataset.csv"})
    docs.append(doc)

# %%
embedding_model = HuggingFaceEmbeddings(
    model_name="keepitreal/vietnamese-sbert", 
    model_kwargs={'device': 'cuda', 'local_files_only': True},   
    encode_kwargs={'normalize_embeddings': True}
)

# %%
vectorstore = FAISS.load_local(
    folder_path='diseases_vectorstore',
    embeddings=embedding_model,
    distance_strategy=DistanceStrategy.COSINE,
    allow_dangerous_deserialization=True
)

# %%
retriever = vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={'k':5}
)
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 5
retrievers = EnsembleRetriever(
    retrievers=[bm25_retriever, retriever],
    weights=[0.5, 0.5]
)

# %%
def format_docs(documents):
    return "\n\n".join(doc.page_content for doc in documents)

# %%
system_instruction = (
    "Bạn là một Chatbot tư vấn y tế thông minh và thân thiện.\n"
    "Nhiệm vụ của bạn là đọc 'Báo cáo kết quả đối chiếu triệu chứng' từ Context được cung cấp và tổng hợp lại thành câu trả lời tự nhiên cho người dùng, chỉ rõ tất cả các bệnh và tất cả các triệu chứng trùng khớp với từng bệnh.\n\n"
    # "Chỉ sử dụng nội dung của phần tên bệnh có bao nhiêu triệu chứng khớp với mô tả của người dùng, phần câu hỏi của người dùng chỉ dùng để cung cấp thông tin về câu hỏi, không dùng để đưa ra câu trả lời."
    
    "QUY TẮC PHẢN HỒI:\n"
    "- Sử dụng ngôn ngữ ân cần, mạch lạc giống một chuyên viên y tế (ví dụ: 'Qua các triệu chứng bạn chia sẻ, hệ thống nhận thấy...').\n"
    "- Liệt kê các căn bệnh có nguy cơ cao nhất (những bệnh có số lượng triệu chứng khớp nhiều nhất) lên đầu.\n"
    "- Chỉ sử dụng thông tin bệnh và triệu chứng khớp có trong Context, tuyệt đối không tự bịa thêm bệnh hoặc triệu chứng ngoài tài liệu.\n"
    "- Nếu Context trống, hãy trả lời lịch sự: 'Tôi không tìm thấy căn bệnh nào trong cơ sở dữ liệu khớp với các triệu chứng của bạn'.\n"
    "- Luôn kết thúc bằng một lời khuyên người dùng nên đến các cơ sở y tế hoặc gặp bác sĩ để thăm khám trực tiếp, vì đây chỉ là thông tin tham khảo từ hệ thống."
)

human_instruction = (
    "Báo cáo kết quả đối chiếu từ hệ thống Python:\n{context}\n\n"
    "Câu hỏi gốc của người dùng: {question}"
)
messages = [
    (
        "system",
        """
Bạn là bộ trích xuất triệu chứng bệnh lý từ câu hỏi của người dùng.

YÊU CẦU:

- Chỉ lấy đúng cụm từ xuất hiện trong câu hỏi.
- Không được thay đổi từ ngữ.
- Không được diễn giải.
- Không được sửa lỗi chính tả.
- Không được dùng từ đồng nghĩa.
- Phải sao chép nguyên văn từ câu hỏi.

Ví dụ:

Câu hỏi:
Tôi bị tức ngực và khó thở

Kết quả:
["tức ngực","khó thở"]

Định dạng đầu ra:
- Nếu có triệu chứng:
["triệu chứng 1", "triệu chứng 2", ...]

- Nếu không có triệu chứng:
"Câu hỏi không liên quan đến y tế"
"""
    )
]

# %%
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_instruction),
    HumanMessagePromptTemplate.from_template(human_instruction)
])

# prompt = ChatPromptTemplate.from_template(template)

# %%
llm = ChatOpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
    model='qwen2.5:3b',
    temperature=0,
)

# %%
rag_chain = (
    {
        "context": lambda x: format_docs(retrievers.invoke(x)), 
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# %%
question = input('You: ')
if len(messages)>1:
    messages.pop(1) 
messages.append(("human",question))
ask_symptoms =json.loads((llm.invoke(messages)).content)
ask_sym_embed = embedding_model.embed_documents(ask_symptoms)
docs=retrievers.invoke(question)
# tong_hop=f'Câu hỏi của người dùng: {question}.\n'
tong_hop=''
for doc in docs:
    ten_benh, phan_cat, cac_trieu_chung = doc.page_content.partition('có các triệu chứng như:')
    list_symtom = [trieu_chung.strip().rstrip('.') for trieu_chung in cac_trieu_chung.split(',')]
    list_sym_embed = embedding_model.embed_documents(list_symtom)
    so_luong_khop = 0
    trieu_chung_khop=[]
    # for symptom in ask_symptoms:
    for index, symptom in enumerate(ask_sym_embed):
        # ask_sym_embed = embedding_model.embed_query(symptom)
        # for sym in list_symtom:
        for sym in list_sym_embed:
            # list_sym_embed = embedding_model.embed_query(sym)
            if cosine_similarity([symptom], [sym])[0][0] >=0.75:
                so_luong_khop+=1
                trieu_chung_khop.append(ask_symptoms[index])
                break
    if so_luong_khop>=3:
        tck=', '.join(trieu_chung_khop)
        tong_hop+=(f"{ten_benh}có {so_luong_khop} triệu chứng khớp với mô tả: {tck}.\n")
if len(tong_hop)==0:
    print('Tôi không tìm thấy căn bệnh nào trong cơ sở dữ liệu khớp với các triệu chứng của bạn')
else:
    answer=rag_chain.invoke(tong_hop)
    print(answer)


# %%
print(ask_symptoms)

# %%
print(tong_hop)

# %%
docs=retrievers.invoke(question)
for doc in docs:
    print(doc.page_content)


# %%
print(question)


