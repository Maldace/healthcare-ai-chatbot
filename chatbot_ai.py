from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama', 
)
context = [
    {
        "role": "system", 
        "content": (
            "Bạn là một trợ lý AI thông minh, chuyên nghiệp và có tư duy logic sắc bén. "
            "Nhiệm vụ của bạn là hỗ trợ người dùng một cách chính xác nhất. "
            "YÊU CẦU BẮT BUỘC: Bạn chỉ được phép suy nghĩ và trả lời hoàn toàn bằng TIẾNG VIỆT tự nhiên. "
            "TUYỆT ĐỐI KHÔNG sử dụng bất kỳ từ ngữ hay ký tự nào thuộc ngôn ngữ khác."
        )
    }
]
while True:
  user_input = input('You:')
  if user_input == 'exit':
    break
  if not user_input.strip():
        continue
  context.append({'role': 'user', 'content': user_input})
  responses_result = client.chat.completions.create(model='qwen2.5:3b', messages=context)
  context.append({'role': 'assistant', 'content': responses_result.choices[0].message.content})
  print(responses_result.choices[0].message.content)