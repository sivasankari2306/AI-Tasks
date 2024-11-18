import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load the pre-trained ResNet model
model = models.resnet50(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define the image transformation pipeline (for preprocessing)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess the image
image_path = 'path_to_image.jpg'  # Replace with the actual image path
img = Image.open(image_path)
img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

# Extract features from the image using ResNet
with torch.no_grad():  # Disable gradient calculation for inference
    features = model(img_tensor)

# The `features` tensor can now be passed to the captioning model


from collections import Counter
from nltk.tokenize import word_tokenize

# Example captions (for training)
captions = [
    "a dog is playing with a ball",
    "a man is running in the field",
    "a cat is sleeping on the couch"
]

# Tokenize the captions and build a vocabulary
tokens = []
for caption in captions:
    tokens += word_tokenize(caption.lower())

# Create a vocabulary with token counts
vocab = Counter(tokens)
vocab_size = len(vocab)

# Create a mapping of words to indices
word_to_index = {word: idx for idx, (word, _) in enumerate(vocab.items())}
index_to_word = {idx: word for word, idx in word_to_index.items()}


import torch.nn as nn
import torch.optim as optim

class CaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(CaptioningModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        embeddings = self.embed(captions)  # Convert words to embeddings
        lstm_input = torch.cat((features.unsqueeze(1), embeddings), dim=1)  # Concatenate image features
        lstm_out, _ = self.lstm(lstm_input)
        output = self.fc(lstm_out)
        return output

# Initialize the model
embed_size = 256  # Size of word embeddings
hidden_size = 512  # Size of LSTM hidden state
vocab_size = vocab_size  # Size of vocabulary
model = CaptioningModel(embed_size, hidden_size, vocab_size)


# Example training loop (simplified)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# For simplicity, assume `features` and `captions` are preprocessed and available
# `features` should be the output from ResNet, `captions` should be tokenized and padded
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Get the model output
    output = model(features, captions)

    # Compute the loss
    loss = criterion(output.view(-1, vocab_size), captions.view(-1))

    # Backpropagate and update the weights
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

def generate_caption(image, model, vocab_size, word_to_index, index_to_word):
    model.eval()
    features = extract_features(image)  # Extract features using ResNet (as done above)

    # Start with a 'start' token (index 0 or another special token)
    input_caption = torch.tensor([word_to_index['start']]).unsqueeze(0)  # Add batch dimension

    generated_caption = []
    for _ in range(max_caption_length):
        output = model(features, input_caption)
        _, predicted_idx = output.max(dim=2)
        predicted_word = index_to_word[predicted_idx.item()]
        generated_caption.append(predicted_word)

        # Stop when 'end' token is generated (index 1 or another special token)
        if predicted_word == 'end':
            break
        
        # Update input caption for the next time step
        input_caption = torch.cat((input_caption, predicted_idx), dim=1)

    return ' '.join(generated_caption)

# Example usage:
image_path = 'path_to_new_image.jpg'
image = Image.open(image_path)
caption = generate_caption(image, model, vocab_size, word_to_index, index_to_word)
print(f"Generated Caption: {caption}")


