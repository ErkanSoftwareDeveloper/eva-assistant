import tkinter as tk  # Tkinter GUI kütüphanesi
from tkinter import scrolledtext  # Kaydırılabilir chat alanı
from transformers import AutoTokenizer, AutoModelForCausalLM  # Hugging Face modeli
import torch  # PyTorch, modeli çalıştırmak için

# -----------------------------
# Profil bilgileri
# -----------------------------
profile = {  # Eva'nın profil bilgileri
    "name": "Eva",  # AI'nin adı
    "favorite_food": "sushi",  # Favori yiyeceği
    "favorite_music": "jazz",  # Favori müzik türü
    "hobbies": ["painting", "reading", "cycling"],  # Hobileri
    "humor_style": "witty and friendly",  # Mizah tarzı
    "favorite_game": "chess"  # Favori oyunu
}

hobbies = ", ".join(profile["hobbies"])  # Listeyi string hâline çevir

# Persona prompt: model cevabı bu profile göre verecek
eva_persona = f"""
You are Eva, an AI assistant. Answer only based on your profile below:
Profile:
- Name: {profile['name']}
- Favorite food: {profile['favorite_food']}
- Favorite music: {profile['favorite_music']}
- Hobbies: {hobbies}
- Humor style: {profile['humor_style']}
- Favorite game: {profile['favorite_game']}

Respond concisely and naturally.
Human: {{user_input}}
Eva: 
"""

# -----------------------------
# Chat modeli
# -----------------------------
model_name = "ericzzz/falcon-rw-1b-chat"  # Kullanılacak model
tokenizer = AutoTokenizer.from_pretrained(model_name)  # Tokenizer yükle
model = AutoModelForCausalLM.from_pretrained(model_name)  # Modeli yükle

# Cihaz ayarı: Mac için MPS, diğerleri CPU
device = torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)  # Modeli cihaza taşı

# -----------------------------
# Tkinter GUI
# -----------------------------
window = tk.Tk()  # Ana pencere
window.title("Eva AI Assistant")  # Pencere başlığı
window.geometry("600x500")  # Pencere boyutu
window.configure(bg="#f0f0f0")  # Arkaplan rengi

# Scrollable chat frame
chat_frame = scrolledtext.ScrolledText(
    window,
    wrap=tk.WORD,  # Kelime kaydırma
    width=70,
    height=25,
    state='disabled',  # Başlangıçta düzenleme kapalı
    bg="#ffffff",
    font=("Helvetica", 11)
)
chat_frame.pack(padx=10, pady=10)

# Input box (kullanıcı mesaj girişi)
input_box = tk.Entry(window, width=60, font=("Helvetica", 12))
input_box.pack(padx=10, pady=5)

# -----------------------------
# Mesaj ekleme fonksiyonu
# -----------------------------


def add_message(text, sender="user"):
    chat_frame.config(state='normal')  # Düzenlemeyi aç
    if sender == "user":
        chat_frame.insert(tk.END, f"You: {text}\n", "user")  # Kullanıcı mesajı
    else:
        chat_frame.insert(tk.END, f"Eva: {text}\n", "eva")  # Eva mesajı
    # Renk ve kenar boşluklarını ayarla
    chat_frame.tag_config("user", foreground="white",
                          background="#4CAF50", spacing3=5, lmargin1=10, lmargin2=10)
    chat_frame.tag_config("eva", foreground="white",
                          background="#2196F3", spacing3=5, lmargin1=10, lmargin2=10)
    chat_frame.config(state='disabled')  # Tekrar düzenlemeyi kapat
    chat_frame.yview(tk.END)  # En son mesaja kaydır

# -----------------------------
# Mesaj gönderme ve /clear desteği
# -----------------------------


def send_message(event=None):
    user_input = input_box.get().strip()  # Kullanıcı mesajını al
    if not user_input:  # Boşsa hiçbir şey yapma
        return

    if user_input.lower() == "/clear":  # /clear komutu
        chat_frame.config(state='normal')
        chat_frame.delete(1.0, tk.END)  # Tüm chat'i temizle
        chat_frame.config(state='disabled')
        input_box.delete(0, tk.END)
        return

    add_message(user_input, sender="user")  # Kullanıcı mesajını ekle
    input_box.delete(0, tk.END)  # Input kutusunu temizle

    # Model prompt'u hazırla
    prompt = eva_persona.format(user_input=user_input)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Cevap üret
    output_ids = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,
    )[0]

    # Sadece yeni kısmı al
    answer_ids = output_ids[inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    add_message(answer, sender="eva")  # Eva cevabını ekle


input_box.bind("<Return>", send_message)  # Enter tuşu mesaj gönderir
input_box.focus()  # Başlangıçta input aktif

window.mainloop()  # GUI başlat
