import re
with open(f"../chatbot_ai_user_and_history/user_and_history/{'Minh'}/{'test'}.txt", "r", encoding="utf-8") as f:
    content = f.read()
chat = re.split(r'(?i)(?=\b(?:user|bot):)', content, flags=re.MULTILINE)
chat = [item.strip() for item in chat if item.strip()]
for text in chat[-10:]:
    # role, msg = text.split(':', 1)
    # if role=='User':
    #     print(msg)
    print(text)