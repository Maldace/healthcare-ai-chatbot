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
from langchain_ollama import ChatOllama
import json

diseases_df = pd.read_csv('./SympScan - Symptomps to Disease/Diseases_and_Symptoms_dataset.csv', encoding='utf-8')

diseases_docs = []
ten_benh = diseases_df['bệnh'].unique()
for benh in ten_benh:
    trieu_chung = []
    cac_mau_benh = diseases_df[diseases_df['bệnh']==benh]
    for i, row in cac_mau_benh.iterrows():
        trieu_chung.extend(col for col in cac_mau_benh.columns if col !='bệnh' and col not in trieu_chung and row[col]==1)
    cac_trieu_chung = ', '.join(trieu_chung)
    chi_tiet_benh = f'{benh} có các triệu chứng như: {cac_trieu_chung}.'
    doc=Document(page_content=chi_tiet_benh,metadata={"source":"Diseases_and_Symptoms_dataset.csv"})
    diseases_docs.append(doc)

description_df = pd.read_csv('SympScan - Symptomps to Disease/description.csv', encoding='utf-8')
description_docs = description_df.astype(str).apply(lambda x: ": ".join(x), axis=1)
description_docs= [Document(page_content=text, metadata={"source":"description.csv"}) for text in description_docs]

diets_df = pd.read_csv('SympScan - Symptomps to Disease/diets.csv', encoding='utf-8')
diets_df['Ăn kiêng'] = diets_df['Ăn kiêng'].astype(str)
diets_docs = []
ten_benh = diets_df['Bệnh'].unique()
for benh in ten_benh:
    che_do_an = ''
    cach_an = diets_df.loc[diets_df['Bệnh']==benh, 'Ăn kiêng'].values[0]
    trans = str.maketrans('', '', "[']")
    che_do_an+=cach_an.translate(trans)
    cac_che_do_an = f'{benh} nên bổ sung: {che_do_an}.'
    cac_che_do_an = cac_che_do_an.replace('\u200b', '')
    doc=Document(page_content=cac_che_do_an,metadata={"source":"diets.csv"})
    diets_docs.append(doc)

medications_df = pd.read_csv('SympScan - Symptomps to Disease/medications.csv', encoding='utf-8')
medications_df['Thuốc'] = medications_df['Thuốc'].astype(str)
medications_docs = []
ten_benh = medications_df['Bệnh'].unique()
for benh in ten_benh:
    thuoc_uong = ''
    thuoc_dung = medications_df.loc[medications_df['Bệnh']==benh, 'Thuốc'].values[0]
    trans = str.maketrans('', '', "[']")
    thuoc_uong+=thuoc_dung.translate(trans)
    cac_thuoc_nen_uong = f'{benh} nên dùng: {thuoc_uong}.'
    cac_thuoc_nen_uong = cac_thuoc_nen_uong.replace('\u200b', '')
    doc=Document(page_content=cac_thuoc_nen_uong,metadata={"source":"medications.csv"})
    medications_docs.append(doc)

precautions_df = pd.read_csv('SympScan - Symptomps to Disease/precautions.csv')
cols = ['Biện pháp phòng ngừa_1', 'Biện pháp phòng ngừa_2', 'Biện pháp phòng ngừa_3', 'Biện pháp phòng ngừa_4']
df = precautions_df[cols].astype(str).agg(', '.join, axis=1)
precautions_docs = []
ten_benh = precautions_df['Bệnh'].unique()
for i, benh in enumerate(ten_benh):
    bien_phap = f'{benh} nên: {df[i]}'
    bien_phap = bien_phap.replace('\u200b', '')
    doc=Document(page_content=bien_phap,metadata={"source":"precautions_.csv"})
    precautions_docs.append(doc)

workout_df = pd.read_csv('SympScan - Symptomps to Disease/workout.csv')
workout_df['Bài tập'] = workout_df['Bài tập'].astype(str)
workout_docs = []
ten_benh = workout_df['Bệnh'].unique()
for benh in ten_benh:
    bai_tap = ''
    bai = workout_df.loc[workout_df['Bệnh']==benh, 'Bài tập'].values[0]
    trans = str.maketrans('', '', '["]')
    bai_tap+=bai.translate(trans)
    cac_bai_tap = f'{benh} nên thực hiện: {bai_tap}.'
    doc=Document(page_content=cac_bai_tap,metadata={"source":"workout.csv"})
    workout_docs.append(doc)

embedding_model = HuggingFaceEmbeddings(
    model_name="keepitreal/vietnamese-sbert", 
    model_kwargs={'device': 'cpu', 'local_files_only': True},   
    encode_kwargs={'normalize_embeddings': True}
)

diseases_vectorstore = FAISS.load_local(
    'diseases_vectorstore',
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)
description_vectorstore = FAISS.load_local(
    'description_vectorstore',
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)
diets_vectorstore = FAISS.load_local(
    'diets_vectorstore',
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)
medications_vectorstore = FAISS.load_local(
    'medications_vectorstore',
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)
precautions_vectorstore = FAISS.load_local(
    'precautions_vectorstore',
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)
workout_vectorstore = FAISS.load_local(
    'workout_vectorstore',
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

def bm25(doc):
    bm25_retriever = BM25Retriever.from_documents(doc)
    bm25_retriever.k = 5
    return bm25_retriever

def format_docs(documents):
    return "\n\n".join(doc.page_content for doc in documents)

def prompt_chat(x, y):
    prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(x),
    HumanMessagePromptTemplate.from_template(y)
    ])
    return prompt

# llm = ChatOpenAI(
#     base_url='http://localhost:11434/v1/',
#     api_key='ollama',
#     model='qwen2.5:3b',
#     temperature=0,
#     model_kwargs={
#         "options": {
#             "num_ctx": 1024,
#             "num_predict": 256,
#             "num_gpu": 0 
#         }
#     }
# )
llm = ChatOllama(
    model="qwen2.5:3b",
    num_ctx=1024,
    num_predict=256,
    num_gpu=0,
    temperature=0
)

def diseases_predict(question):
    retriever = diseases_vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={'k':5}
    )

    retrievers = EnsembleRetriever(
    retrievers=[bm25(diseases_docs), retriever],
    weights=[0.5, 0.5]
    )
    
    system_instruction = (
    "Bạn là một Chatbot tư vấn y tế thông minh và thân thiện.\n"
    "Nhiệm vụ của bạn là đọc 'Báo cáo kết quả đối chiếu triệu chứng' từ Context được cung cấp và tổng hợp lại thành câu trả lời tự nhiên cho người dùng, chỉ rõ tất cả các bệnh và tất cả các triệu chứng trùng khớp với từng bệnh.\n\n"
    
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

    Định dạng đầu ra PHẢI LÀ một JSON array:
    - Nếu có triệu chứng:
    ["triệu chứng 1", "triệu chứng 2", ...]

    - Nếu không có triệu chứng:
    "Câu hỏi không liên quan đến y tế"
    """
        ),
        ("human", question)
    ]

    rag_chain = (
    {
        "context": lambda x: format_docs(retrievers.invoke(x)), 
        "question": RunnablePassthrough()
    }
    | prompt_chat(system_instruction, human_instruction)
    | llm
    | StrOutputParser()
    )
    # rag_chain = (
    # prompt_chat(system_instruction, human_instruction)
    # | llm
    # | StrOutputParser()
    # )

    ask_symptoms =json.loads((llm.invoke(messages)).content)
    if ask_symptoms == 'Câu hỏi không liên quan đến y tế':
        print(ask_symptoms)
        return
    ask_symptoms = [text.lower() for text in ask_symptoms]
    ask_sym_embed = embedding_model.embed_documents(ask_symptoms)
    docs=retrievers.invoke(question)
    tong_hop=''
    for doc in docs:
        ten_benh, phan_cat, cac_trieu_chung = doc.page_content.partition('có các triệu chứng như:')
        cac_trieu_chung = cac_trieu_chung.lower()
        list_symtom = [trieu_chung.strip().rstrip('.') for trieu_chung in cac_trieu_chung.split(',')]
        list_sym_embed = embedding_model.embed_documents(list_symtom)
        so_luong_khop = 0
        trieu_chung_khop=[]
        for index, symptom in enumerate(ask_sym_embed):
            for sym in list_sym_embed:
                if cosine_similarity([symptom], [sym])[0][0] >=0.75:
                    so_luong_khop+=1
                    trieu_chung_khop.append(ask_symptoms[index])
                    break
        if so_luong_khop>=3:
            tck=', '.join(trieu_chung_khop)
            tong_hop+=(f"{ten_benh}có {so_luong_khop} triệu chứng khớp với mô tả: {tck}.\n")
    if len(tong_hop)==0:
        print('Hệ thống không tìm thấy căn bệnh nào trong cơ sở dữ liệu khớp với các triệu chứng của bạn')
    else:
        answer=rag_chain.invoke(tong_hop)
        # answer=rag_chain.invoke({'context':tong_hop, 'question':question})
        print(answer)

def ask_description(question):
    retriever = description_vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={'k':5}
    )

    retrievers = EnsembleRetriever(
    retrievers=[bm25(description_docs), retriever],
    weights=[0.5, 0.5]
    )
    
    system_instruction = (
    # "Bạn là một Chatbot tư vấn y tế thông minh và thân thiện.\n"
    # "Nhiệm vụ của bạn là đọc 'Báo cáo mô tả chứng bệnh' từ Context được cung cấp và tổng hợp lại thành câu trả lời tự nhiên cho người dùng, chỉ rõ mô tả về tất cả các căn bệnh của người dùng có thể đang gặp phải.\n\n"
    
    # "QUY TẮC PHẢN HỒI:\n"
    # "- Sử dụng ngôn ngữ ân cần, mạch lạc giống một chuyên viên y tế (ví dụ: 'Căn bệnh mà bạn đang muốn tìm hiểu là...').\n"
    # "- Không nói rằng không tìm thấy dữ liệu."
    # "- Chỉ sử dụng thông tin bệnh và triệu chứng khớp có trong Context, tuyệt đối không tự bịa thêm thông tin ngoài tài liệu.\n"
    # "- Luôn kết thúc bằng một lời khuyên người dùng nên đến các cơ sở y tế hoặc gặp bác sĩ để thăm khám trực tiếp, vì đây chỉ là thông tin tham khảo từ hệ thống."
    """
    Bạn là một Chatbot tư vấn y tế thông minh và thân thiện.

    Dưới đây là danh sách các bệnh và mô tả.

    Hãy diễn đạt lại thành câu trả lời tự nhiên, chỉ rõ mô tả về tất cả các căn bệnh của người dùng có thể đang gặp phải.

    QUY TẮC PHẢN HỒI:
    - Sử dụng ngôn ngữ ân cần, mạch lạc giống một chuyên viên y tế (ví dụ: 'Căn bệnh mà bạn đang muốn tìm hiểu là...').
    - Không thêm, không bớt, không suy luận, không nói rằng không tìm thấy dữ liệu.
    - Triển khai mỗi ý nằm ổ một dòng, không đưa câu trả lời lên một dòng duy nhất
    - Luôn kết thúc bằng một lời khuyên người dùng nên đến các cơ sở y tế hoặc gặp bác sĩ để thăm khám trực tiếp, vì đây chỉ là thông tin tham khảo từ hệ thống.
    """
    )

    human_instruction = (
        "Báo cáo kết quả đối chiếu từ hệ thống Python:\n{context}\n\n"
        "Câu hỏi gốc của người dùng: {question}"
    )
    messages = [
        (
            "system",
            """
    Bạn là bộ trích xuất tên bệnh lý từ câu hỏi của người dùng.

    YÊU CẦU:

    - Chỉ lấy đúng cụm từ về căn bệnh xuất hiện trong câu hỏi.
    - Không được thay đổi từ ngữ.
    - Không được diễn giải.
    - Không được sửa lỗi chính tả.
    - Không được dùng từ đồng nghĩa.
    - Phải sao chép nguyên văn từ câu hỏi.

    Ví dụ:

    Câu hỏi:
    Bệnh cảm lạnh thông thường và dị ứng là bệnh gì?

    Kết quả:
    ["cảm lạnh thông thường", "dị ứng"]

    Định dạng đầu ra PHẢI LÀ một JSON array:
    - Nếu có tên bệnh:
    ["tên bệnh 1", "tên bệnh 2", ...]

    - Nếu không có tên bệnh:
    "Câu hỏi không liên quan đến y tế"
    """
        ),
        ("human", question)
    ]

    # rag_chain = (
    # {
    #     "context": lambda x: format_docs(retrievers.invoke(x)), 
    #     "question": RunnablePassthrough()
    # }
    # | prompt_chat(system_instruction, human_instruction)
    # | llm
    # | StrOutputParser()
    # )
    rag_chain = (
    prompt_chat(system_instruction, human_instruction)
    | llm
    | StrOutputParser()
    )

    ask_diseases =json.loads((llm.invoke(messages)).content)
    if ask_diseases == 'Câu hỏi không liên quan đến y tế':
        print(ask_diseases)
        return
    ask_diseases = [text.lower().strip() for text in ask_diseases]
    # ask_dis_embed = embedding_model.embed_documents(ask_diseases)
    tong_hop=''
    for diseases in ask_diseases:
        ask_dis_embed = embedding_model.embed_query(diseases)
        docs=retrievers.invoke(diseases)
    # docs=retrievers.invoke(question)
        for doc in docs:
            ten_benh = ((doc.page_content).split(':', 1)[0]).lower()
            if ten_benh.startswith("bệnh "):
                ten_benh = ten_benh[5:]
            dis_embed = embedding_model.embed_query(ten_benh.lower().strip())
            # for dis in ask_dis_embed:
            #     if cosine_similarity([dis], [dis_embed])[0][0] >=0.75:
            #         tong_hop += f'{doc.page_content}\n'
            if cosine_similarity([ask_dis_embed], [dis_embed])[0][0] >=0.95:
                tong_hop += f'{doc.page_content}\n'
    if len(tong_hop)==0:
        print('Hệ thống không tìm thấy căn bệnh nào trong cơ sở dữ liệu khớp với các triệu chứng của bạn')
        return
    # print(format_docs(retrievers.invoke(tong_hop)))
    answer=rag_chain.invoke({'context':tong_hop, 'question':question})
    print(answer)

def ask_diets(question):
    retriever = diets_vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={'k':5}
    )

    retrievers = EnsembleRetriever(
    retrievers=[bm25(diets_docs), retriever],
    weights=[0.5, 0.5]
    )
    
    system_instruction = (
    # "Bạn là một Chatbot tư vấn y tế thông minh và thân thiện.\n"
    # "Nhiệm vụ của bạn là đọc 'Báo cáo mô tả chứng bệnh' từ Context được cung cấp và tổng hợp lại thành câu trả lời tự nhiên cho người dùng, chỉ rõ mô tả về tất cả các căn bệnh của người dùng có thể đang gặp phải.\n\n"
    
    # "QUY TẮC PHẢN HỒI:\n"
    # "- Sử dụng ngôn ngữ ân cần, mạch lạc giống một chuyên viên y tế (ví dụ: 'Căn bệnh mà bạn đang muốn tìm hiểu là...').\n"
    # "- Không nói rằng không tìm thấy dữ liệu."
    # "- Chỉ sử dụng thông tin bệnh và triệu chứng khớp có trong Context, tuyệt đối không tự bịa thêm thông tin ngoài tài liệu.\n"
    # "- Luôn kết thúc bằng một lời khuyên người dùng nên đến các cơ sở y tế hoặc gặp bác sĩ để thăm khám trực tiếp, vì đây chỉ là thông tin tham khảo từ hệ thống."
    """
    Bạn là một Chatbot tư vấn y tế thông minh và thân thiện.

    Dưới đây là danh sách các bệnh và mô tả.

    Hãy diễn đạt lại thành câu trả lời tự nhiên, chỉ rõ mô tả về tất cả các căn bệnh của người dùng có thể đang gặp phải.

    QUY TẮC PHẢN HỒI:
    - Sử dụng ngôn ngữ ân cần, mạch lạc giống một chuyên viên y tế (ví dụ: 'Căn bệnh mà bạn đang muốn tìm hiểu là...').
    - Không thêm, không bớt, không suy luận, không nói rằng không tìm thấy dữ liệu.
    - Triển khai mỗi ý nằm ổ một dòng, không đưa câu trả lời lên một dòng duy nhất
    - Luôn kết thúc bằng một lời khuyên người dùng nên đến các cơ sở y tế hoặc gặp bác sĩ để thăm khám trực tiếp, vì đây chỉ là thông tin tham khảo từ hệ thống.
    """
    )

    human_instruction = (
        "Báo cáo kết quả đối chiếu từ hệ thống Python:\n{context}\n\n"
        "Câu hỏi gốc của người dùng: {question}"
    )
    messages = [
        (
            "system",
            """
    Bạn là bộ trích xuất tên bệnh lý từ câu hỏi của người dùng.

    YÊU CẦU:

    - Chỉ lấy đúng cụm từ về căn bệnh xuất hiện trong câu hỏi.
    - Không được thay đổi từ ngữ.
    - Không được diễn giải.
    - Không được sửa lỗi chính tả.
    - Không được dùng từ đồng nghĩa.
    - Phải sao chép nguyên văn từ câu hỏi.

    Ví dụ:

    Câu hỏi:
    Bệnh cảm lạnh thông thường và dị ứng nên ăn gì?

    Kết quả:
    ["cảm lạnh thông thường", "dị ứng"]

    Định dạng đầu ra PHẢI LÀ một JSON array:
    - Nếu có tên bệnh:
    ["tên bệnh 1", "tên bệnh 2", ...]

    - Nếu không có tên bệnh:
    "Câu hỏi không liên quan đến y tế"
    """
        ),
        ("human", question)
    ]

    # rag_chain = (
    # {
    #     "context": lambda x: format_docs(retrievers.invoke(x)), 
    #     "question": RunnablePassthrough()
    # }
    # | prompt_chat(system_instruction, human_instruction)
    # | llm
    # | StrOutputParser()
    # )
    rag_chain = (
    prompt_chat(system_instruction, human_instruction)
    | llm
    | StrOutputParser()
    )

    ask_diseases =json.loads((llm.invoke(messages)).content)
    if ask_diseases == 'Câu hỏi không liên quan đến y tế':
        print(ask_diseases)
        return
    ask_diseases = [text.lower().strip() for text in ask_diseases]
    # ask_dis_embed = embedding_model.embed_documents(ask_diseases)
    tong_hop=''
    for diseases in ask_diseases:
        ask_dis_embed = embedding_model.embed_query(diseases)
        docs=retrievers.invoke(diseases)
    # docs=retrievers.invoke(question)
        for doc in docs:
            ten_benh = ((doc.page_content).split('nên bổ sung:', 1)[0]).lower()
            if ten_benh.startswith("bệnh "):
                ten_benh = ten_benh[5:]
            dis_embed = embedding_model.embed_query(ten_benh.lower().strip())
            # for dis in ask_dis_embed:
            #     if cosine_similarity([dis], [dis_embed])[0][0] >=0.75:
            #         tong_hop += f'{doc.page_content}\n'
            if cosine_similarity([ask_dis_embed], [dis_embed])[0][0] >=0.75:
                tong_hop += f'{doc.page_content}\n'
    if len(tong_hop)==0:
        print('Hệ thống không tìm thấy căn bệnh nào trong cơ sở dữ liệu khớp với các triệu chứng của bạn')
        return
    # print(format_docs(retrievers.invoke(tong_hop)))
    answer=rag_chain.invoke({'context':tong_hop, 'question':question})
    print(answer)

def ask_medications(question):
    retriever = medications_vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={'k':5}
    )

    retrievers = EnsembleRetriever(
    retrievers=[bm25(medications_docs), retriever],
    weights=[0.5, 0.5]
    )
    
    system_instruction = (
    # "Bạn là một Chatbot tư vấn y tế thông minh và thân thiện.\n"
    # "Nhiệm vụ của bạn là đọc 'Báo cáo mô tả chứng bệnh' từ Context được cung cấp và tổng hợp lại thành câu trả lời tự nhiên cho người dùng, chỉ rõ mô tả về tất cả các căn bệnh của người dùng có thể đang gặp phải.\n\n"
    
    # "QUY TẮC PHẢN HỒI:\n"
    # "- Sử dụng ngôn ngữ ân cần, mạch lạc giống một chuyên viên y tế (ví dụ: 'Căn bệnh mà bạn đang muốn tìm hiểu là...').\n"
    # "- Không nói rằng không tìm thấy dữ liệu."
    # "- Chỉ sử dụng thông tin bệnh và triệu chứng khớp có trong Context, tuyệt đối không tự bịa thêm thông tin ngoài tài liệu.\n"
    # "- Luôn kết thúc bằng một lời khuyên người dùng nên đến các cơ sở y tế hoặc gặp bác sĩ để thăm khám trực tiếp, vì đây chỉ là thông tin tham khảo từ hệ thống."
    """
    Bạn là một Chatbot tư vấn y tế thông minh và thân thiện.

    Dưới đây là danh sách các bệnh và mô tả.

    Hãy diễn đạt lại thành câu trả lời tự nhiên, chỉ rõ mô tả về tất cả các căn bệnh của người dùng có thể đang gặp phải.

    QUY TẮC PHẢN HỒI:
    - Sử dụng ngôn ngữ ân cần, mạch lạc giống một chuyên viên y tế (ví dụ: 'Căn bệnh mà bạn đang muốn tìm hiểu là...').
    - Không thêm, không bớt, không suy luận, không nói rằng không tìm thấy dữ liệu.
    - Triển khai mỗi ý nằm ổ một dòng, không đưa câu trả lời lên một dòng duy nhất
    - Luôn kết thúc bằng một lời khuyên người dùng nên đến các cơ sở y tế hoặc gặp bác sĩ để thăm khám trực tiếp, vì đây chỉ là thông tin tham khảo từ hệ thống.
    """
    )

    human_instruction = (
        "Báo cáo kết quả đối chiếu từ hệ thống Python:\n{context}\n\n"
        "Câu hỏi gốc của người dùng: {question}"
    )
    messages = [
        (
            "system",
            """
    Bạn là bộ trích xuất tên bệnh lý từ câu hỏi của người dùng.

    YÊU CẦU:

    - Chỉ lấy đúng cụm từ về căn bệnh xuất hiện trong câu hỏi.
    - Không được thay đổi từ ngữ.
    - Không được diễn giải.
    - Không được sửa lỗi chính tả.
    - Không được dùng từ đồng nghĩa.
    - Phải sao chép nguyên văn từ câu hỏi.

    Ví dụ:

    Câu hỏi:
    Bệnh cảm lạnh thông thường và dị ứng nên dùng thuốc gì?

    Kết quả:
    ["cảm lạnh thông thường", "dị ứng"]

    Định dạng đầu ra PHẢI LÀ một JSON array:
    - Nếu có tên bệnh:
    ["tên bệnh 1", "tên bệnh 2", ...]

    - Nếu không có tên bệnh:
    "Câu hỏi không liên quan đến y tế"
    """
        ),
        ("human", question)
    ]

    # rag_chain = (
    # {
    #     "context": lambda x: format_docs(retrievers.invoke(x)), 
    #     "question": RunnablePassthrough()
    # }
    # | prompt_chat(system_instruction, human_instruction)
    # | llm
    # | StrOutputParser()
    # )
    rag_chain = (
    prompt_chat(system_instruction, human_instruction)
    | llm
    | StrOutputParser()
    )

    ask_diseases =json.loads((llm.invoke(messages)).content)
    if ask_diseases == 'Câu hỏi không liên quan đến y tế':
        print(ask_diseases)
        return
    ask_diseases = [text.lower().strip() for text in ask_diseases]
    # ask_dis_embed = embedding_model.embed_documents(ask_diseases)
    tong_hop=''
    for diseases in ask_diseases:
        ask_dis_embed = embedding_model.embed_query(diseases)
        docs=retrievers.invoke(diseases)
    # docs=retrievers.invoke(question)
        for doc in docs:
            ten_benh = ((doc.page_content).split('nên dùng:', 1)[0]).lower()
            if ten_benh.startswith("bệnh "):
                ten_benh = ten_benh[5:]
            dis_embed = embedding_model.embed_query(ten_benh.lower().strip())
            # for dis in ask_dis_embed:
            #     if cosine_similarity([dis], [dis_embed])[0][0] >=0.75:
            #         tong_hop += f'{doc.page_content}\n'
            if cosine_similarity([ask_dis_embed], [dis_embed])[0][0] >=0.75:
                tong_hop += f'{doc.page_content}\n'
    if len(tong_hop)==0:
        print('Hệ thống không tìm thấy căn bệnh nào trong cơ sở dữ liệu khớp với các triệu chứng của bạn')
        return
    # print(format_docs(retrievers.invoke(tong_hop)))
    answer=rag_chain.invoke({'context':tong_hop, 'question':question})
    print(answer)

def ask_precautions(question):
    retriever = precautions_vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={'k':5}
    )

    retrievers = EnsembleRetriever(
    retrievers=[bm25(precautions_docs), retriever],
    weights=[0.5, 0.5]
    )
    
    system_instruction = (
    # "Bạn là một Chatbot tư vấn y tế thông minh và thân thiện.\n"
    # "Nhiệm vụ của bạn là đọc 'Báo cáo mô tả chứng bệnh' từ Context được cung cấp và tổng hợp lại thành câu trả lời tự nhiên cho người dùng, chỉ rõ mô tả về tất cả các căn bệnh của người dùng có thể đang gặp phải.\n\n"
    
    # "QUY TẮC PHẢN HỒI:\n"
    # "- Sử dụng ngôn ngữ ân cần, mạch lạc giống một chuyên viên y tế (ví dụ: 'Căn bệnh mà bạn đang muốn tìm hiểu là...').\n"
    # "- Không nói rằng không tìm thấy dữ liệu."
    # "- Chỉ sử dụng thông tin bệnh và triệu chứng khớp có trong Context, tuyệt đối không tự bịa thêm thông tin ngoài tài liệu.\n"
    # "- Luôn kết thúc bằng một lời khuyên người dùng nên đến các cơ sở y tế hoặc gặp bác sĩ để thăm khám trực tiếp, vì đây chỉ là thông tin tham khảo từ hệ thống."
    """
    Bạn là một Chatbot tư vấn y tế thông minh và thân thiện.

    Dưới đây là danh sách các bệnh và mô tả.

    Hãy diễn đạt lại thành câu trả lời tự nhiên, chỉ rõ mô tả về tất cả các căn bệnh của người dùng có thể đang gặp phải.

    QUY TẮC PHẢN HỒI:
    - Sử dụng ngôn ngữ ân cần, mạch lạc giống một chuyên viên y tế (ví dụ: 'Căn bệnh mà bạn đang muốn tìm hiểu là...').
    - Không thêm, không bớt, không suy luận, không nói rằng không tìm thấy dữ liệu.
    - Triển khai mỗi ý nằm ổ một dòng, không đưa câu trả lời lên một dòng duy nhất
    - Luôn kết thúc bằng một lời khuyên người dùng nên đến các cơ sở y tế hoặc gặp bác sĩ để thăm khám trực tiếp, vì đây chỉ là thông tin tham khảo từ hệ thống.
    """
    )

    human_instruction = (
        "Báo cáo kết quả đối chiếu từ hệ thống Python:\n{context}\n\n"
        "Câu hỏi gốc của người dùng: {question}"
    )
    messages = [
        (
            "system",
            """
    Bạn là bộ trích xuất tên bệnh lý từ câu hỏi của người dùng.

    YÊU CẦU:

    - Chỉ lấy đúng cụm từ về căn bệnh xuất hiện trong câu hỏi.
    - Không được thay đổi từ ngữ.
    - Không được diễn giải.
    - Không được sửa lỗi chính tả.
    - Không được dùng từ đồng nghĩa.
    - Phải sao chép nguyên văn từ câu hỏi.

    Ví dụ:

    Câu hỏi:
    Bệnh cảm lạnh thông thường và dị ứng nên làm gì?

    Kết quả:
    ["cảm lạnh thông thường", "dị ứng"]

    Định dạng đầu ra PHẢI LÀ một JSON array:
    - Nếu có tên bệnh:
    ["tên bệnh 1", "tên bệnh 2", ...]

    - Nếu không có tên bệnh:
    "Câu hỏi không liên quan đến y tế"
    """
        ),
        ("human", question)
    ]

    # rag_chain = (
    # {
    #     "context": lambda x: format_docs(retrievers.invoke(x)), 
    #     "question": RunnablePassthrough()
    # }
    # | prompt_chat(system_instruction, human_instruction)
    # | llm
    # | StrOutputParser()
    # )
    rag_chain = (
    prompt_chat(system_instruction, human_instruction)
    | llm
    | StrOutputParser()
    )

    ask_diseases =json.loads((llm.invoke(messages)).content)
    # print(ask_diseases)
    if ask_diseases == 'Câu hỏi không liên quan đến y tế':
        print(ask_diseases)
        return
    ask_diseases = [text.lower().strip() for text in ask_diseases]
    # ask_dis_embed = embedding_model.embed_documents(ask_diseases)
    tong_hop=''
    for diseases in ask_diseases:
        ask_dis_embed = embedding_model.embed_query(diseases)
        docs=retrievers.invoke(diseases)
    # docs=retrievers.invoke(question)
        for doc in docs:
            ten_benh = ((doc.page_content).split('nên:', 1)[0]).lower()
            if ten_benh.startswith("bệnh "):
                ten_benh = ten_benh[5:]
            dis_embed = embedding_model.embed_query(ten_benh.lower().strip())
            # for dis in ask_dis_embed:
            #     if cosine_similarity([dis], [dis_embed])[0][0] >=0.75:
            #         tong_hop += f'{doc.page_content}\n'
            if cosine_similarity([ask_dis_embed], [dis_embed])[0][0] >=0.75:
                tong_hop += f'{doc.page_content}\n'
    if len(tong_hop)==0:
        print('Hệ thống không tìm thấy căn bệnh nào trong cơ sở dữ liệu khớp với các triệu chứng của bạn')
        return
    # print(format_docs(retrievers.invoke(tong_hop)))
    answer=rag_chain.invoke({'context':tong_hop, 'question':question})
    print(answer)

def ask_workout(question):
    retriever = workout_vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={'k':5}
    )

    retrievers = EnsembleRetriever(
    retrievers=[bm25(workout_docs), retriever],
    weights=[0.5, 0.5]
    )
    
    system_instruction = (
    # "Bạn là một Chatbot tư vấn y tế thông minh và thân thiện.\n"
    # "Nhiệm vụ của bạn là đọc 'Báo cáo mô tả chứng bệnh' từ Context được cung cấp và tổng hợp lại thành câu trả lời tự nhiên cho người dùng, chỉ rõ mô tả về tất cả các căn bệnh của người dùng có thể đang gặp phải.\n\n"
    
    # "QUY TẮC PHẢN HỒI:\n"
    # "- Sử dụng ngôn ngữ ân cần, mạch lạc giống một chuyên viên y tế (ví dụ: 'Căn bệnh mà bạn đang muốn tìm hiểu là...').\n"
    # "- Không nói rằng không tìm thấy dữ liệu."
    # "- Chỉ sử dụng thông tin bệnh và triệu chứng khớp có trong Context, tuyệt đối không tự bịa thêm thông tin ngoài tài liệu.\n"
    # "- Luôn kết thúc bằng một lời khuyên người dùng nên đến các cơ sở y tế hoặc gặp bác sĩ để thăm khám trực tiếp, vì đây chỉ là thông tin tham khảo từ hệ thống."
    """
    Bạn là một Chatbot tư vấn y tế thông minh và thân thiện.

    Dưới đây là danh sách các bệnh và mô tả.

    Hãy diễn đạt lại thành câu trả lời tự nhiên, chỉ rõ mô tả về tất cả các căn bệnh của người dùng có thể đang gặp phải.

    QUY TẮC PHẢN HỒI:
    - Sử dụng ngôn ngữ ân cần, mạch lạc giống một chuyên viên y tế (ví dụ: 'Căn bệnh mà bạn đang muốn tìm hiểu là...').
    - Không thêm, không bớt, không suy luận, không nói rằng không tìm thấy dữ liệu.
    - Triển khai mỗi ý nằm ổ một dòng, không đưa câu trả lời lên một dòng duy nhất
    - Luôn kết thúc bằng một lời khuyên người dùng nên đến các cơ sở y tế hoặc gặp bác sĩ để thăm khám trực tiếp, vì đây chỉ là thông tin tham khảo từ hệ thống.
    """
    )

    human_instruction = (
        "Báo cáo kết quả đối chiếu từ hệ thống Python:\n{context}\n\n"
        "Câu hỏi gốc của người dùng: {question}"
    )
    messages = [
        (
            "system",
            """
    Bạn là bộ trích xuất tên bệnh lý từ câu hỏi của người dùng.

    YÊU CẦU:

    - Chỉ lấy đúng cụm từ về căn bệnh xuất hiện trong câu hỏi.
    - Không được thay đổi từ ngữ.
    - Không được diễn giải.
    - Không được sửa lỗi chính tả.
    - Không được dùng từ đồng nghĩa.
    - Phải sao chép nguyên văn từ câu hỏi.

    Ví dụ:

    Câu hỏi:
    Bệnh cảm lạnh thông thường và dị ứng nên luyện tập gì?

    Kết quả:
    ["cảm lạnh thông thường", "dị ứng"]

    Định dạng đầu ra PHẢI LÀ một JSON array:
    - Nếu có tên bệnh:
    ["tên bệnh 1", "tên bệnh 2", ...]

    - Nếu không có tên bệnh:
    "Câu hỏi không liên quan đến y tế"
    """
        ),
        ("human", question)
    ]

    # rag_chain = (
    # {
    #     "context": lambda x: format_docs(retrievers.invoke(x)), 
    #     "question": RunnablePassthrough()
    # }
    # | prompt_chat(system_instruction, human_instruction)
    # | llm
    # | StrOutputParser()
    # )
    rag_chain = (
    prompt_chat(system_instruction, human_instruction)
    | llm
    | StrOutputParser()
    )

    ask_diseases =json.loads((llm.invoke(messages)).content)
    # print(ask_diseases)
    if ask_diseases == 'Câu hỏi không liên quan đến y tế':
        print(ask_diseases)
        return
    ask_diseases = [text.lower().strip() for text in ask_diseases]
    # ask_dis_embed = embedding_model.embed_documents(ask_diseases)
    tong_hop=''
    for diseases in ask_diseases:
        ask_dis_embed = embedding_model.embed_query(diseases)
        docs=retrievers.invoke(diseases)
    # docs=retrievers.invoke(question)
        for doc in docs:
            ten_benh = ((doc.page_content).split('nên thực hiện:', 1)[0]).lower()
            if ten_benh.startswith("bệnh "):
                ten_benh = ten_benh[5:]
            dis_embed = embedding_model.embed_query(ten_benh.lower().strip())
            # for dis in ask_dis_embed:
            #     if cosine_similarity([dis], [dis_embed])[0][0] >=0.75:
            #         tong_hop += f'{doc.page_content}\n'
            if cosine_similarity([ask_dis_embed], [dis_embed])[0][0] >=0.75:
                tong_hop += f'{doc.page_content}\n'
    if len(tong_hop)==0:
        print('Hệ thống không tìm thấy căn bệnh nào trong cơ sở dữ liệu khớp với các triệu chứng của bạn')
        return
    # print(format_docs(retrievers.invoke(tong_hop)))
    answer=rag_chain.invoke({'context':tong_hop, 'question':question})
    print(answer)

def ask_symptom(question):
    retriever = diseases_vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={'k':5}
    )

    retrievers = EnsembleRetriever(
    retrievers=[bm25(diseases_docs), retriever],
    weights=[0.5, 0.5]
    )
    
    system_instruction = (
    # "Bạn là một Chatbot tư vấn y tế thông minh và thân thiện.\n"
    # "Nhiệm vụ của bạn là đọc 'Báo cáo mô tả chứng bệnh' từ Context được cung cấp và tổng hợp lại thành câu trả lời tự nhiên cho người dùng, chỉ rõ mô tả về tất cả các căn bệnh của người dùng có thể đang gặp phải.\n\n"
    
    # "QUY TẮC PHẢN HỒI:\n"
    # "- Sử dụng ngôn ngữ ân cần, mạch lạc giống một chuyên viên y tế (ví dụ: 'Căn bệnh mà bạn đang muốn tìm hiểu là...').\n"
    # "- Không nói rằng không tìm thấy dữ liệu."
    # "- Chỉ sử dụng thông tin bệnh và triệu chứng khớp có trong Context, tuyệt đối không tự bịa thêm thông tin ngoài tài liệu.\n"
    # "- Luôn kết thúc bằng một lời khuyên người dùng nên đến các cơ sở y tế hoặc gặp bác sĩ để thăm khám trực tiếp, vì đây chỉ là thông tin tham khảo từ hệ thống."
    """
    Bạn là một Chatbot tư vấn y tế thông minh và thân thiện.

    Dưới đây là danh sách các bệnh và mô tả.

    Hãy diễn đạt lại thành câu trả lời tự nhiên, chỉ rõ mô tả về tất cả các căn bệnh của người dùng có thể đang gặp phải.

    QUY TẮC PHẢN HỒI:
    - Sử dụng ngôn ngữ ân cần, mạch lạc giống một chuyên viên y tế (ví dụ: 'Căn bệnh mà bạn đang muốn tìm hiểu là...').
    - Không thêm, không bớt, không suy luận, không nói rằng không tìm thấy dữ liệu.
    - Triển khai mỗi ý nằm ổ một dòng, không đưa câu trả lời lên một dòng duy nhất
    - Luôn kết thúc bằng một lời khuyên người dùng nên đến các cơ sở y tế hoặc gặp bác sĩ để thăm khám trực tiếp, vì đây chỉ là thông tin tham khảo từ hệ thống.
    """
    )

    human_instruction = (
        "Báo cáo kết quả đối chiếu từ hệ thống Python:\n{context}\n\n"
        "Câu hỏi gốc của người dùng: {question}"
    )
    messages = [
        (
            "system",
            """
    Bạn là bộ trích xuất tên bệnh lý từ câu hỏi của người dùng.

    YÊU CẦU:

    - Chỉ lấy đúng cụm từ về căn bệnh xuất hiện trong câu hỏi.
    - Không được thay đổi từ ngữ.
    - Không được diễn giải.
    - Không được sửa lỗi chính tả.
    - Không được dùng từ đồng nghĩa.
    - Phải sao chép nguyên văn từ câu hỏi.

    Ví dụ:

    Câu hỏi:
    Bệnh cảm lạnh thông thường và dị ứng có các triệu chứng gì?

    Kết quả:
    ["cảm lạnh thông thường", "dị ứng"]

    Định dạng đầu ra PHẢI LÀ một JSON array:
    - Nếu có tên bệnh:
    ["tên bệnh 1", "tên bệnh 2", ...]

    - Nếu không có tên bệnh:
    "Câu hỏi không liên quan đến y tế"
    """
        ),
        ("human", question)
    ]

    # rag_chain = (
    # {
    #     "context": lambda x: format_docs(retrievers.invoke(x)), 
    #     "question": RunnablePassthrough()
    # }
    # | prompt_chat(system_instruction, human_instruction)
    # | llm
    # | StrOutputParser()
    # )
    rag_chain = (
    prompt_chat(system_instruction, human_instruction)
    | llm
    | StrOutputParser()
    )

    ask_diseases =json.loads((llm.invoke(messages)).content)
    # print(ask_diseases)
    if ask_diseases == 'Câu hỏi không liên quan đến y tế':
        print(ask_diseases)
        return
    ask_diseases = [text.lower().strip() for text in ask_diseases]
    # ask_dis_embed = embedding_model.embed_documents(ask_diseases)
    tong_hop=''
    for diseases in ask_diseases:
        ask_dis_embed = embedding_model.embed_query(diseases)
        docs=retrievers.invoke(diseases)
    # docs=retrievers.invoke(question)
        for doc in docs:
            ten_benh = ((doc.page_content).split('có các triệu chứng như:', 1)[0]).lower()
            if ten_benh.startswith("bệnh "):
                ten_benh = ten_benh[5:]
            dis_embed = embedding_model.embed_query(ten_benh.lower().strip())
            # for dis in ask_dis_embed:
            #     if cosine_similarity([dis], [dis_embed])[0][0] >=0.75:
            #         tong_hop += f'{doc.page_content}\n'
            if cosine_similarity([ask_dis_embed], [dis_embed])[0][0] >=0.75:
                tong_hop += f'{doc.page_content}\n'
    if len(tong_hop)==0:
        print('Hệ thống không tìm thấy căn bệnh nào trong cơ sở dữ liệu khớp với các triệu chứng của bạn')
        return
    # print(format_docs(retrievers.invoke(tong_hop)))
    answer=rag_chain.invoke({'context':tong_hop, 'question':question})
    print(answer)

question = input('You: ')
if ('bị bệnh' in question) or ('bị' and 'bệnh gì' in question) or ('bị gì' in question):
    diseases_predict(question)
elif 'là' in question or 'bệnh' in question:
    ask_description(question)
elif 'ăn' in question or 'bổ sung' in question or 'dùng' in question:
    ask_diets(question)
elif 'thuốc' in question or 'uống' in question:
    ask_medications(question)
elif 'làm' in question or 'phòng' in question or 'ngừa' in question:
    ask_precautions(question)
elif 'tập' in question or 'luyện' in question:
    ask_workout(question)
elif 'triệu chứng' in question:
    ask_symptom(question)
else:
    print('Câu hỏi không liên quan đến y tế')


