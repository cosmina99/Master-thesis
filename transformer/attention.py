import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torchvision
import torchvision.transforms as T
from torchvision.models.feature_extraction import create_feature_extractor

img = Image.open('../data/images/MET/214/3.jpg')

img = T.Resize((224,224))(T.ToTensor()(img))
plt.imshow(img.permute(1, 2, 0))
plt.show()

model = torchvision.models.vit_b_16(pretrained=True)
feature_extractor = create_feature_extractor(model, return_nodes=['encoder.layers.encoder_layer_11.ln'])
feature_extractor.eval()

output = feature_extractor(img.unsqueeze(0))

x = output['encoder.layers.encoder_layer_11.ln']
x_, attn = model.encoder.layers[-1].self_attention(x, x, x, need_weights=True, average_attn_weights=False)

for i in range(12):
    sns.heatmap(attn[0, i, 0, 1:].view(14, 14).detach().numpy())
    plt.show()