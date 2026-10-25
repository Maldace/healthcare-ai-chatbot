from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_classic.vectorstores import FAISS
# from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_ollama import ChatOllama
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.passthrough import RunnablePassthrough
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from sentence_transformers import CrossEncoder
import numpy as np
import json

with open('document.txt', encoding='utf-8') as f:
    text = f.read()
    
headers_to_split_on = [
    ("#", "Header")
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
chunks = markdown_splitter.split_text(text)

embedding_model = HuggingFaceEmbeddings(
    model_name="keepitreal/vietnamese-sbert", 
    model_kwargs={'device': 'cuda', 'local_files_only': True},   
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = FAISS.load_local(
    'vectorstore/document_vectorstore',
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

llm = ChatOllama(
    model="qwen2.5:3b",
    num_ctx=4096,
    num_predict=2048,
    num_gpu=0,
    temperature=0
)

# chat = HuggingFaceEndpoint(
#     repo_id='Qwen/Qwen2.5-3B',
#     task="text-generation",
#     max_new_tokens=2048,
# )
# llm = ChatHuggingFace(llm=chat)

reranking_model = CrossEncoder(
    "Qwen/Qwen3-Reranker-0.6B",
    prompts={'instruction':
    """
    Bạn là công cụ sắp xếp câu hỏi và tài liệu về y tế.
    Hãy sắp xếp các cặp câu hỏi và tài liệu được đưa theo thứ tự giảm dần, ưu tiên tài liệu liên quan nhất đến câu hỏi.
    """},
    default_prompt_name='instruction'            
    )

def prompt_chat(x, y):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(x),
        HumanMessagePromptTemplate.from_template(y)
    ])
    return prompt

def retriever():
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 20
    vectorstore_retriever = vectorstore.as_retriever(
        search_type = 'similarity',
        search_kwargs = {'k':20}
    )
    retrievers = EnsembleRetriever(
        retrievers = [bm25_retriever, vectorstore_retriever],
        weights = [0.3, 0.7]
    )
    # chat = 'cách phòng ngừa bệnh addison'
    # print('bm25')
    # bm25 = bm25_retriever.invoke(chat)
    # for num, i in enumerate(bm25):
    #    print(num)
    #    print(i.page_content)
    #    print('===========')
    
    # print('vec')
    # vec = vectorstore_retriever.invoke(chat)
    # for num,  i in enumerate(vec):
    #     print(num)
    #     print(i.page_content)
    #     print('===========')
    # bm25 = retrievers.invoke(chat)
    # for num, i in enumerate(bm25):
    #     print(num)
    #     print(i.page_content)
    #     print('===========')
    return retrievers

def chat_with_bot(question, chat, user):
    system_instruction = (
        "Bạn là một Chatbot tư vấn y tế thông minh và thân thiện.\n"
        "Nhiệm vụ của bạn là đọc thông tin được cung cấp để diễn đạt lại thành câu trả lời tự nhiên cho người dùng về vấn đề họ gặp phải.\n\n"
        
        "QUY TẮC PHẢN HỒI:\n"
        "- Sử dụng ngôn ngữ ân cần, mạch lạc giống một chuyên viên y tế (ví dụ: 'Qua các thông tin bạn chia sẻ, hệ thống nhận thấy...').\n"
        "- Chỉ đưa ra câu trả lời dựa trên tài liệu được cung cấp, không bớt, không suy luận thêm, không dùng kiến thức đã được huấn luyện sẵn từ trước để trả lời.\n"
        "- Triển khai mỗi ý nằm ổ một dòng, không đưa câu trả lời lên một dòng duy nhất.\n"
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

    # query = 'tôi bị tiêu chảy, buồn nôn, đau bụng, đầy hơi là bệnh gì'
    
    ask_diseases =json.loads((llm.invoke(messages)).content)
    if ask_diseases == 'Câu hỏi không liên quan đến y tế':
        return ask_diseases
    ask_diseases = [text.lower().strip() for text in ask_diseases]
    with open(f"../chatbot_ai_user_and_history/user_and_history/{user}/{chat}.json", "r", encoding='utf-8') as f:
        json_data = json.load(f)
    json_data['last_diseases'] = ask_diseases
    json_str = json.dumps(json_data, ensure_ascii=False, indent=4)
    with open(f"../chatbot_ai_user_and_history/user_and_history/{user}/{chat}.json", "w", encoding='utf-8') as f:
        f.write(json_str)

    docs = retriever().invoke(question)

    def format_docs(documents):
        return "\n\n".join(doc.page_content for doc in documents)

    pairs = [(question, doc.page_content) for doc in docs]
    score = reranking_model.predict(pairs)

    rag_chain = (
        # {'context':lambda _: format_docs(docs), 'question': RunnablePassthrough()}
        {'context':lambda _: format_docs([docs[np.where(score==np.max(score))[0][0]]]), 'question': RunnablePassthrough()}
        | prompt_chat(system_instruction, human_instruction)
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(question)
    print()
    # print(docs[np.where(score==np.max(score))[0][0]])
    return answer


# retriever()
# answer = chat_with_bot('cách phòng ngừa bệnh addison', 'test', 'Minh')
# print('=============')
# print(answer)