import matplotlib.pyplot as plt
import numpy as np
import cifar10loader
import matplotlib

print(matplotlib.get_backend())

def plot(i):
    train_loader,test_loader,train_dataset,test_dataset = cifar10loader.loadcifar10()
    img = cifar10loader.loadImageI(i,test_dataset)  # This is your custom function returning a torch Tensor
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])

    # Copy the tensor so we don't mutate the original
    img_for_plot = img.clone()
    img_for_plot = img_for_plot.cpu()

    # 2) Now that unnormalization is done, permute *once* from [C,H,W] -> [H,W,C]
    img_for_plot = img_for_plot.permute(1, 2, 0)

    # 3) Convert it to a NumPy array
    img_for_plot = img_for_plot.numpy()

    # 4) Show the image
    plt.imshow(img_for_plot)
    plt.title(f"Label: image {i}")
    plt.axis('off')
    plt.savefig("my_image.png") 
plot(3)