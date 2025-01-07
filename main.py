import os
import discord
from discord.ext import commands
from PIL import Image
import requests
import torch
from torchvision import models, transforms
import io

# Charger le fichier .env (si vous utilisez des variables d'environnement)
# load_dotenv()

# Récupérer le token à partir de la variable d'environnement
token = os.getenv('MTMyNDkwMzg0NTAyOTA4OTM3MQ.GA66CA.igEYHIah_rxug1h8XrROdE4magXGgB56uKgM3Y')

# Vérification du token
if not token:
    print("Erreur : Le token Discord n'a pas été trouvé.")
    exit(1)

# Activer les intents nécessaires pour le bot Discord
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Charger un modèle pré-entraîné (ResNet par exemple)
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()  # Mode évaluation

# Transformation de l'image pour correspondre au modèle
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Fonction pour analyser une image et retourner une prédiction
def analyze_image(image_url):
    try:
        print(f"Téléchargement de l'image depuis {image_url}")
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")

        # Prétraiter l'image
        input_tensor = preprocess(image).unsqueeze(0)

        # Effectuer la prédiction
        with torch.no_grad():
            output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)

        # Retourner la classe prédite
        return f"Classe prédite : {predicted_class.item()}"
    except Exception as e:
        print(f"Erreur lors de l'analyse de l'image: {e}")
        return f"Erreur lors de l'analyse : {e}"

# Événement de démarrage du bot
@bot.event
async def on_ready():
    print(f"Bot connecté en tant que {bot.user}")

# Commande de test pour vérifier la réactivité du bot
@bot.command(name="ping")
async def ping(ctx):
    await ctx.send("Pong!")

# Commande pour analyser une image envoyée par l'utilisateur
@bot.command(name="recherche")
async def recherche(ctx):
    await ctx.send("Veuillez envoyer l'image à analyser.")

    def check(m):
        return m.author == ctx.author and len(m.attachments) > 0

    try:
        # Attendre que l'utilisateur envoie une image
        message = await bot.wait_for("message", check=check, timeout=30.0)
        image_url = message.attachments[0].url

        # Analyser l'image
        result = analyze_image(image_url)

        # Envoyer le résultat au canal Discord
        await ctx.send(result)
    except Exception as e:
        await ctx.send(f"Erreur : {e}")

# Démarrer le bot
bot.run(token)
