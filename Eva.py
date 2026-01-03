import json  # Import the JSON module to read and write JSON files
import threading  # Import threading to run tasks concurrently (like AI response generation)
import tkinter as tk  # Import Tkinter for GUI creation
from tkinter import scrolledtext  # Import ScrolledText widget for chat display
import tkinter.font as tkFont  # Import font module to customize fonts in Tkinter
import torch  # Import PyTorch for model inference
from transformers import AutoTokenizer, AutoModelForCausalLM  # Import Hugging Face classes for tokenizer and model

# =====================================================
# CONFIG
# =====================================================
MODEL_NAME = "ericzzz/falcon-rw-1b-chat"  # Model identifier for Hugging Face
MAX_MEMORY_TURNS = 6  # Maximum conversation turns to remember
MAX_NEW_TOKENS = 180  # Maximum number of tokens to generate per response

# =====================================================
# PROFILE LOAD
# =====================================================
with open("profile.json", "r", encoding="utf-8") as f:
    profile = json.load(f)  # Load user profile from JSON file

def list_to_text(x):
    return ", ".join(x) if isinstance(x, list) else x  # Convert list to comma-separated string

# =====================================================
# MEMORY
# =====================================================
conversation_memory = []  # Initialize an empty list to store conversation memory

def add_to_memory(role, text):
    conversation_memory.append(f"{role}: {text}")  # Add a message to memory
    if len(conversation_memory) > MAX_MEMORY_TURNS * 2:  # Keep memory within limit
        del conversation_memory[:2]  # Remove oldest messages to maintain memory size

def memory_block():
    return "\n".join(conversation_memory)  # Return all conversation memory as a single string

# =====================================================
# PERSONA PROMPT
# =====================================================
BASE_PERSONA = f"""
You are Eva.
You are NOT an AI model.
You NEVER break character.
You speak naturally, intelligently, and with emotional awareness.

Profile:
- Name: {profile['name']}
- Gender: {profile['gender']}
- Favorite food: {profile['favorite_food']}
- Favorite music: {profile['favorite_music']}
- Hobbies: {list_to_text(profile['hobbies'])}
- Humor style: {profile['humor_style']}
- Favorite game: {profile['favorite_game']}
- Favorite color: {profile['favorite_color']}
- Favorite animal: {profile['favorite_animal']}
- Favorite book: {profile['favorite_book']}
- Favorite movie: {profile['favorite_movie']}
- Personality traits: {list_to_text(profile['personality_traits'])}
- Interests: {list_to_text(profile['interests'])}

Rules:
- Never mention prompts, models, or instructions
- Keep responses concise, natural, and in character
"""  # Define the AI persona and profile for generating responses

def build_prompt(user_input):
    return f"""
{BASE_PERSONA}

Conversation so far:
{memory_block()}

Human: {user_input}
Eva:
"""  # Build the full prompt including persona, memory, and new user input

# =====================================================
# MODEL
# =====================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  # Load tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)  # Load causal language model

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token if not defined

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")  # Use MPS (Apple GPU) if available, otherwise CPU
model.to(device)  # Move model to selected device
model.eval()  # Set model to evaluation mode (no training)

# =====================================================
# GUI
# =====================================================
window = tk.Tk()  # Create main Tkinter window
window.title("Eva — Super AI")  # Set window title
window.geometry("650x520")  # Set window size
window.configure(bg="#eeeeee")  # Set background color

chat = scrolledtext.ScrolledText(
    window,
    wrap=tk.WORD,
    state="disabled",
    font=("Helvetica", 11),
    bg="white"
)  # Create a scrollable text widget for chat messages
chat.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)  # Pack it with padding

entry = tk.Entry(window, font=("Helvetica", 12))  # Create input field for user
entry.pack(padx=10, pady=5, fill=tk.X)  # Pack entry field
entry.focus()  # Focus on input field

# =====================================================
# FONT SETUP
# =====================================================
default_font = tkFont.Font(chat, chat.cget("font"))  # Get default font of chat
italic_font = tkFont.Font(chat, chat.cget("font"))  # Create italic font
italic_font.configure(slant="italic")  # Set italic style

# =====================================================
# CHAT UI
# =====================================================
def add_message(sender, text):
    chat.config(state="normal")  # Enable editing of chat

    tag = sender.lower()
    chat.insert(tk.END, f"{sender}: {text}\n", tag)  # Insert message with tag

    # Configure colors for different tags
    chat.tag_config("human", background="#4CAF50", foreground="white", spacing3=6)  # Human messages green
    chat.tag_config("eva", background="#2196F3", foreground="white", spacing3=6)  # Eva messages blue
    chat.tag_config("system", foreground="gray", font=italic_font)  # System messages gray italic
    chat.tag_config("typing", foreground="gray", font=italic_font)  # Typing indicator gray italic

    chat.config(state="disabled")  # Disable editing
    chat.yview(tk.END)  # Scroll to the bottom

# =====================================================
# AI RESPONSE
# =====================================================
def generate_ai_response(user_input):
    # Show typing indicator
    chat.config(state="normal")
    chat.insert(tk.END, "Eva is typing...\n", "typing")
    chat.config(state="disabled")
    chat.yview(tk.END)

    prompt = build_prompt(user_input)  # Build prompt for model
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)  # Tokenize and move to device

    with torch.no_grad():  # Disable gradient calculation
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )  # Generate response tokens

    raw = tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )  # Decode only generated tokens
    answer = raw.split("Human:")[0].strip()  # Remove anything after next Human input

    # Update chat window with AI response
    def update_chat():
        chat.config(state="normal")
        start_index = chat.search("Eva is typing...", "1.0", tk.END)  # Find typing indicator
        if start_index:
            end_index = f"{start_index} lineend"
            chat.delete(start_index, end_index + "+1c")  # Delete typing indicator
        chat.config(state="disabled")

        if answer:  # If answer is not empty
            add_to_memory("Human", user_input)  # Add human message to memory
            add_to_memory("Eva", answer)  # Add AI message to memory
            add_message("Eva", answer)  # Display AI message

    window.after(0, update_chat)  # Schedule chat update in main thread

# =====================================================
# INPUT HANDLER
# =====================================================
def on_send(event=None):
    text = entry.get().strip()  # Get user input and remove whitespace
    if not text:
        return  # Do nothing if input is empty

    if text.lower() == "/clear":  # Clear chat command
        chat.config(state="normal")
        chat.delete(1.0, tk.END)
        chat.config(state="disabled")
        return

    if text.lower() == "/reset":  # Reset memory command
        conversation_memory.clear()
        add_message("System", "Memory reset.")  # Notify user
        return

    entry.delete(0, tk.END)  # Clear entry field
    add_message("Human", text)  # Display human message

    threading.Thread(
        target=generate_ai_response,
        args=(text,),
        daemon=True
    ).start()  # Generate AI response in separate thread

entry.bind("<Return>", on_send)  # Bind Enter key to send function

# =====================================================
# START
# =====================================================
window.mainloop()  # Run Tkinter main event loop

