import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.label import MDLabel
from kivy.clock import Clock
from kivy.core.window import Window

# --- АРХИТЕКТУРА ТВОЕЙ МОДЕЛИ ---
class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 128)
        self.pos_embed = nn.Parameter(torch.zeros(1, 16, 128))
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(128, vocab_size)

    def forward(self, x):
        T = x.shape[1]
        x = self.embed(x) + self.pos_embed[:, :T, :]
        mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1)
        x = self.transformer(x, mask=mask)
        return self.fc(x)

# --- ДИЗАЙН ИНТЕРФЕЙСА (KV) ---
KV = '''
MDScreen:
    md_bg_color: 0.05, 0.05, 0.05, 1
    MDBoxLayout:
        orientation: 'vertical'

        MDTopAppBar:
            title: "ЯРИК AI v3.0"
            elevation: 4
            md_bg_color: 0.1, 0.1, 0.1, 1
            specific_text_color: 0, 1, 0, 1
            right_action_items: [["robot", lambda x: app.toggle_yar()]]

        ScrollView:
            id: scroll
            MDBoxLayout:
                id: chat_box
                orientation: 'vertical'
                adaptive_height: True
                padding: "15dp"
                spacing: "15dp"

        MDBoxLayout:
            size_hint_y: None
            height: "70dp"
            padding: "10dp"
            spacing: "10dp"
            md_bg_color: 0.1, 0.1, 0.1, 1

            MDTextField:
                id: user_input
                hint_text: "Напиши Ярику..."
                mode: "round"
                fill_color_normal: 0.15, 0.15, 0.15, 1
                text_color_normal: 1, 1, 1, 1
                on_text_validate: app.send_message()

            MDIconButton:
                icon: "send"
                theme_text_color: "Custom"
                text_color: 0, 1, 0, 1
                on_release: app.send_message()
'''

class YarikApp(MDApp):
    yar_mode = True  # Режим Ярика по умолчанию
    replacements = {"ты": "ти", "тебе": "тибе", "тебя": "тибя", "привет": "ку", "что": "шо"}
    phrases = ["жужалица", "жижа", "буль буль", "я сгорел"]

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Green"
        self.load_model()
        return Builder.load_string(KV)

    def load_model(self):
        # Автоматический поиск файлов в папке с приложением
        base = os.path.dirname(__file__)
        data_path = os.path.join(base, "data_marked.txt")
        model_path = os.path.join(base, "gpt_model_marked.pt")
        
        with open(data_path, "r", encoding="utf-8") as f:
            words = f.read().lower().split()
        self.vocab = sorted(set(words))
        self.w2i = {w: i for i, w in enumerate(self.vocab)}
        self.i2w = {i: w for w, i in self.w2i.items()}
        
        self.model = GPT(len(self.vocab))
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()

    def toggle_yar(self):
        self.yar_mode = not self.yar_mode
        status = "ВКЛ" if self.yar_mode else "ВЫКЛ"
        self.add_msg(f"--- Режим Ярика: {status} ---", (0.5, 0.5, 0.5, 1))

    def send_message(self):
        text = self.root.ids.user_input.text.strip()
        if text:
            self.add_msg(f"Ты: {text}", (1, 1, 1, 1))
            self.root.ids.user_input.text = ""
            Clock.schedule_once(lambda dt: self.generate_reply(text), 0.5)

    def add_msg(self, text, color):
        lbl = MDLabel(
            text=text, 
            theme_text_color="Custom", 
            text_color=color, 
            size_hint_y=None, 
            adaptive_height=True
        )
        self.root.ids.chat_box.add_widget(lbl)
        self.root.ids.scroll.scroll_y = 0

    def generate_reply(self, text):
        # Логика генерации
        tokens = [self.w2i.get(w, 0) for w in text.lower().split()[-14:]]
        if not tokens: tokens = [0]
        tokens_tensor = torch.tensor([tokens], dtype=torch.long)
        
        with torch.no_grad():
            logits = self.model(tokens_tensor[:, -16:])
            probs = F.softmax(logits[0, -1] / 0.8, dim=0)
            next_token = torch.multinomial(probs, 1).item()
        
        reply = self.i2w.get(next_token, "жужалица")
        
        # Стилизация Ярика
        if self.yar_mode:
            words = reply.split()
            styled = [self.replacements.get(w, w) for w in words]
            if random.random() < 0.2: styled.append(random.choice(self.phrases))
            reply = " ".join(styled)

        self.add_msg(f"Ярик: {reply}", (0, 1, 0, 1))

if __name__ == "__main__":
    YarikApp().run()
