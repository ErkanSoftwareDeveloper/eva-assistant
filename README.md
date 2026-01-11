# 🤖 Eva — AI Chat Assistant

A conversational **AI chat desktop application** built with **Python**, **Tkinter**, and **Hugging Face Transformers**.
Eva remembers recent conversations, responds naturally, and simulates a personalized AI companion.

---

## ✨ Features

* Chat with Eva in a user-friendly GUI
* Memory of the last few conversation turns
* Personalized responses based on a user profile
* Typing indicator while Eva is generating a response
* Commands:

  * `/clear` — clear chat window
  * `/reset` — reset conversation memory
* Color-coded messages for human and Eva
* Threaded AI response generation for smooth UI

---

## 🛠️ Technologies Used

* **Python 3**
* **Tkinter** – GUI framework
* **PyTorch** – for model inference
* **Transformers (Hugging Face)** – AI model and tokenizer
* **JSON** – storing user profile and conversation memory
* **Threading** – responsive UI during AI computation

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/ErkanSoftwareDeveloper/eva-ai-chat.git
cd eva-ai-chat
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Make sure you have **PyTorch** installed with a compatible version for your system.

---

## ▶️ Usage

Run the application:

```bash
python Eva.py
```

1. Type a message and press **Enter** to send
2. Use `/clear` to clear the chat window
3. Use `/reset` to reset Eva’s memory
4. Chat with Eva and watch her respond naturally

---

## 📁 Project Structure

```text
eva-ai-chat/
├─ Eva.py              # Main application file
├─ profile.json        # User profile for personalized responses
├─ .gitignore          # Git ignored files
├─ README.md           # Project documentation
└─ requirements.txt    # Project dependencies
```

---

## 📸 Video

![2026-01-1111-26-57-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/edbb6829-84e2-4c3f-a7d7-aa74a52ebf54)


---

## 🚀 Possible Improvements

* Voice input/output
* Conversation history saving/loading
* Multi-profile support
* Dark/light theme UI
* Packaging as executable (.exe)

---

## 📄 License

This project is intended for **educational and personal use**.
