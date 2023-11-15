import tkinter as tk
from tkinter import scrolledtext
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
import transformers
import threading

class ChatbotGUI:
    def __init__(self, master):
        self.master = master
        master.title("Felix Chatbot GUI")

        # Create a scrolled text area for the conversation
        self.conversation_text = scrolledtext.ScrolledText(master, width=60, height=20, wrap=tk.WORD)
        self.conversation_text.pack()

        # Create an entry for user input
        self.input_entry = tk.Entry(master, width=50)
        self.input_entry.pack()

        # Create a button to send user input
        self.send_button = tk.Button(master, text="Send", command=self.send_user_input)
        self.send_button.pack()

        # Create a text widget to display conversation history
        self.history_text = tk.Text(master, height=10, width=60)
        self.history_text.pack()

        # Initialize the chatbot components
        self.initialize_chatbot()

    def initialize_chatbot(self):
        # Loading documents
        loader = UnstructuredFileLoader("./data.txt")
        documents = loader.load()

        # Text splitting
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0,
            separator="\n\n",
            length_function=len
        )
        text = text_splitter.split_documents(documents)

        # Huggingface Embeddings and Vector Store
        embedding = HuggingFaceEmbeddings()
        vectordb = Chroma.from_documents(text, embedding=embedding)

        # Model and Huggingface pipeline
        model_name = 'declare-lab/flan-alpaca-base'
        generate_text = transformers.pipeline(
            model=model_name,
            task='text2text-generation',
            max_length=1100,
            temperature=0.9,
            repetition_penalty=1.1
        )
        llm = HuggingFacePipeline(pipeline=generate_text)

        # Memory and Conversational Chain
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        self.chatbot = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectordb.as_retriever(), memory=memory)

        # Welcome Message
        self.display_message("Felix Chatbot: Welcome! Ask me anything or type 'exit' to end the conversation.")

    def send_user_input(self):
        # Get user input
        user_input = self.input_entry.get()

        # Display user input in the conversation area
        self.display_message(f"You: {user_input}", user_input=True)

        # Check for exit command
        if user_input.lower() == 'exit':
            self.display_message("Felix Chatbot: Goodbye!", chatbot=True)
            self.master.quit()
        else:
            # Create a thread to run the chatbot processing in the background
            threading.Thread(target=self.process_user_input, args=(user_input,)).start()

            # Clear the input entry
            self.input_entry.delete(0, tk.END)

    def process_user_input(self, user_input):
        # Get and display chatbot response
        response = self.chatbot.run(user_input)
        self.display_message(f"Felix Chatbot: {response}", chatbot=True)

        # Update conversation history
        self.update_history(f"You: {user_input}\nFelix Chatbot: {response}\n\n")

    def display_message(self, message, user_input=False, chatbot=False):
        # Configure tag and insert the message into the conversation area
        tag = "user" if user_input else "chatbot" if chatbot else None
        self.conversation_text.insert(tk.END, f"\n{message}\n", tag)

        # Apply formatting (color, font, etc.) based on the tag
        if tag == "user":
            self.conversation_text.tag_config(tag, foreground="blue", font=("Helvetica", 10, "bold"))
        elif tag == "chatbot":
            self.conversation_text.tag_config(tag, foreground="green", font=("Courier", 10, "italic"))

        # Scroll to the end of the conversation area
        self.conversation_text.see(tk.END)

    def update_history(self, message):
        # Insert the message into the conversation history
        self.history_text.insert(tk.END, message)

        # Scroll to the end of the conversation history
        self.history_text.see(tk.END)

def main():
    root = tk.Tk()
    chatbot_gui = ChatbotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()