import torchvision
import torchvision.transforms as transforms

mnist = torchvision.datasets.MNIST(download=False, train=True, root="./data")

data_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize((mnist.data.float().mean() / 255, ),
                         (mnist.data.float().std() / 255, ))
])