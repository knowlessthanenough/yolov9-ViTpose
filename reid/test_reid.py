import os
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import torchreid

# Setup
image_folder = './reid_test_image/team'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 32

# Load pretrained OSNet
model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=1000,
    pretrained=True
)
model.eval().to(device)

# Preprocessing transform
transform = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# Collect image paths
image_paths = [os.path.join(image_folder, f)
               for f in os.listdir(image_folder)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

features = []
filenames = []

# Batch inference loop
for i in tqdm(range(0, len(image_paths), batch_size)):
    batch_paths = image_paths[i:i + batch_size]
    batch_imgs = []

    for path in batch_paths:
        img = Image.open(path).convert('RGB')
        img_tensor = transform(img)
        batch_imgs.append(img_tensor)

    batch_tensor = torch.stack(batch_imgs).to(device)  # shape [B, 3, 256, 128]

    with torch.no_grad():
        batch_features = model(batch_tensor)  # shape [B, 512]

    features.append(batch_features.cpu())  # move to CPU to save memory
    filenames.extend(batch_paths)

# Combine all
features = torch.cat(features, dim=0)  # [N, 512]
print(f"✅ Extracted features for {features.shape[0]} crops.")

# Optional: Save as dict
torch.save({'filenames': filenames, 'features': features}, 'features.pt')
print("✅ Saved features to features.pt")
