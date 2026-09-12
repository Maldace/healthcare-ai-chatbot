import re
with open(f"user_and_history/User/General.txt", "r", encoding="utf-8") as f:
    content = f.read()
    chat = re.split(r'(?i)(?=\b(?:user|bot):)', content, flags=re.MULTILINE)
    chat = [item.strip() for item in chat if item.strip()]
    # print(chat)
    for text in chat:
        sender, msg = text.split(":", 1)
        print(sender)
# txt = "hello, my name is Peter, I am 26 years old"

# x, y = txt.split(", ", 1)

# print(y)