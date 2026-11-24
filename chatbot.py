from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_classic.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from sentence_transformers import CrossEncoder
import numpy as np
import json
import re
import time

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
    model="qwen3:4b-instruct",
    num_ctx=1024,
    num_predict=256,
    temperature=0,
    reasoning = False,
    keep_alive='30m',
    num_gpu=0
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
    default_prompt_name='instruction',
    device='cuda'       
    )

def prompt_chat(x, y):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(x),
        HumanMessagePromptTemplate.from_template(y)
    ])
    return prompt

def retriever():
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 100
    vectorstore_retriever = vectorstore.as_retriever(
        search_type = 'similarity',
        search_kwargs = {'k':100}
    )
    retrievers = EnsembleRetriever(
        retrievers = [bm25_retriever, vectorstore_retriever],
        weights = [0.3, 0.7]
    )
    return retrievers

def format_docs(documents):
    return "\n\n".join(doc.page_content for doc in documents)

def get_answer(question):
    system_instruction = (
        "Bạn là một Chatbot tư vấn y tế thông minh và thân thiện.\n"
        "Nhiệm vụ của bạn là đọc thông tin được cung cấp để diễn đạt lại thành câu trả lời tự nhiên cho người dùng về vấn đề họ gặp phải bằng tiếng Việt.\n\n"
        
        
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
    docs = retriever().invoke(question)

    pairs = [(question, doc.page_content) for doc in docs]
    score = reranking_model.predict(pairs)

    rag_chain = (
        {'context':lambda _: format_docs([docs[np.where(score==np.max(score))[0][0]]]), 'question': RunnablePassthrough()}
        | prompt_chat(system_instruction, human_instruction)
        | llm
        | StrOutputParser()
    )

    # print(pairs)
    # print(score)

    print(docs[np.where(score==np.max(score))[0][0]])
    answer = rag_chain.invoke(question)
    return answer

store = {}

def get_by_session_id(chat: str, user: str)-> BaseChatMessageHistory:
    if (chat, user) not in store:
        store[(chat, user)] = InMemoryChatMessageHistory()
        with open(f"../chatbot_ai_user_and_history/user_and_history/{user}/{chat}.txt", "r", encoding="utf-8") as f:
            content = f.read()
        chat_history = re.split(r'(?i)(?=\b(?:user|bot):)', content, flags=re.MULTILINE)
        chat_history = [item.strip() for item in chat_history if item.strip()]
        for text in chat_history[-10:]:
            role, msg = text.split(':', 1)
            if role=='User':
                store[(chat, user)].add_user_message(msg)
            else:
                store[(chat, user)].add_ai_message(msg)
    return store[(chat, user)]

def chat(question, chat, user):
    # start = time.time()
    prompt_text =(
    """
    Bạn là hệ thống viết lại câu hỏi về y tế dựa trên lịch sử trò chuyện.

    **NHIỆM VỤ:**

    Tạo một câu hỏi độc lập từ câu hỏi mới nhất của người dùng bằng tiếng Việt.

    **QUY TẮC BẮT BUỘC:**

    1. Trước tiên, xác định câu hỏi mới nhất có liên quan đến lĩnh vực y tế hay không.

    2. Nếu câu hỏi liên quan đến y tế, tiếp tục thực hiện nhiệm vụ viết lại câu hỏi. Không được kết luận câu hỏi không liên quan đến y tế chỉ vì nó không đề cập đến chủ đề trong lịch sử trò chuyện.

    3. Đọc lịch sử để xác định chủ đề hoặc đối tượng mà người dùng đang đề cập đến.

    4. Nếu câu hỏi chứa các cụm từ tham chiếu như: "nó", "bệnh này", "bệnh đó", "loại bệnh trên", hãy thay thế các cụm từ tham chiếu đó bằng đối tượng cụ thể đã được đề cập trong lịch sử gần nhất có liên quan.

    5. Ưu tiên đối tượng được người dùng hỏi đến trong các lượt hội thoại trước đó.

    6. TUYỆT ĐỐI KHÔNG tự tạo tên bệnh hoặc đối tượng không xuất hiện trong lịch sử.

    7. Không sử dụng kiến thức bên ngoài để suy đoán đối tượng mà người dùng đang đề cập.

    8. Nếu không xác định chắc chắn đối tượng được nhắc đến, hãy giữ nguyên câu hỏi ban đầu.

    9. Nếu câu hỏi đề cập đến một bệnh hoặc chủ đề y tế mới, không phụ thuộc vào lịch sử trò chuyện, hãy giữ nguyên câu hỏi ban đầu.

    10. Giữ nguyên mục đích và ý nghĩa câu hỏi. Không trả lời câu hỏi.

    11. Nếu câu hỏi không liên quan đến y tế, trả lời chính xác: Câu hỏi không liên quan đến y tế.

    **ĐẦU RA:**

    Chỉ trả về câu hỏi được viết lại hoặc chuỗi "Câu hỏi không liên quan đến y tế". Không giải thích thêm.
    """    
    )
    
    messages = [
        (
            "system",
            """
    Bạn là bộ trích xuất triệu chứng bệnh lý và tên bệnh từ câu hỏi của người dùng.
    
    YÊU CẦU:

    - Chỉ lấy đúng cụm từ xuất hiện trong câu hỏi.
    - Không được thay đổi từ ngữ.
    - Không được diễn giải.
    - Không được sửa lỗi chính tả.
    - Không được dùng từ đồng nghĩa.
    - Phải sao chép nguyên văn từ câu hỏi.

    Ví dụ:

    Câu hỏi:
    Tôi bị tức ngực và khó thở là bệnh gì?

    Kết quả:
    ["tức ngực","khó thở"]
    
    ------------
    
    Câu hỏi:
    Tiểu đường là bệnh gì?

    Kết quả:
    ["tiểu đường"]

    Định dạng đầu ra PHẢI LÀ một JSON array:
    - Nếu có triệu chứng hoặc tên bệnh:
    ["triệu chứng 1", "triệu chứng 2", ...]
    """
        ),
        ("human", question)
    ]
    
    # ask_diseases =(llm.invoke(messages)).content
    # ask_diseases =json.loads((llm.invoke(messages)).content)
    # if ask_diseases == 'Câu hỏi không liên quan đến y tế':
    #     return ask_diseases
    if (re.search(r"\bbị bệnh\b", question) or (re.search(r"\bbị\b", question) and re.search(r"\bbệnh gì\b", question)) or re.search(r"\bbị gì\b", question)):
        ask_diseases =json.loads((llm.invoke(messages)).content)
        if len(ask_diseases) < 3:
            return 'Không đủ dữ kiện để kết luận, vui lòng cung cấp từ 3 triệu chứng trở lên'
        else:
            return get_answer(question)
            # return len(ask_diseases)
    # return question
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_text),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    
    chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_by_session_id,
        input_messages_key="question",
        history_messages_key="history",
        history_factory_config=[
            ConfigurableFieldSpec(
                id="chat",
                annotation=str,
                name="Chat name",
                description="Unique identifier for the chat.",
                default="",
                is_shared=True,
            ),
            ConfigurableFieldSpec(
                id="user",
                annotation=str,
                name="Username",
                description="Unique identifier for the user.",
                default="",
                is_shared=True,
            ),
        ],
    )
    
    # return(chain_with_history.invoke({'question':question}, config = {"configurable": { "chat": chat, "user": user}}))
    # return(chain_with_history.invoke({'question':question}, config = {"configurable": { "session_id": "foo"}}))
    rewrite_question = chain_with_history.invoke({'question':question}, config = {"configurable": {"chat": chat, "user": user}})
    # time.sleep(1)
    # end = time.time()

    # print(f"Total runtime of the program is {end - start} seconds")
    if rewrite_question == 'Câu hỏi không liên quan đến y tế.':
        return rewrite_question
    return get_answer(rewrite_question)



# print(chat('ù tai có triệu chứng gì', 'test', 'Minh'))

# import time

# start = time.time()


# print(chat('bệnh đó có triệu chứng gì', 'test', 'Minh'))

# time.sleep(1)
# end = time.time()

# print(f"Total runtime of the program is {end - start} seconds")
# print(chat('bệnh trĩ là gì', '123', 'Phuc'))