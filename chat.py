with open(f"user_and_history/abc.txt", "x", encoding="utf-8"):
    pass

general="""User:Welcome to the Healthcare Chatbot AI!
Bot:Here you can ask the chatbot about symptoms, medicine, diet,... of some diseases.
User:Be careful, as the information given by chatbot can be miscorrect and is just for reference only, cannot replace doctor's instruction.
Bot:So, enjoy!!
        """

with open(f"user_and_history/abc.txt", "a", encoding="utf-8") as f:
                f.write(general)