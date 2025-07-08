import random
import matplotlib.pyplot as plt

index = random.randint(0, len(test_dataset))
image, label = test_dataset[index]

plt.imshow(image.squeeze(), cmap='gray')
plt.title(f'Actual Label: {label}')
plt.show()

image = image.view(1, 28*28)
output = model(image)
_, predicted = torch.max(output, 1)
print(f'Predicted Label: {predicted.item()}')
