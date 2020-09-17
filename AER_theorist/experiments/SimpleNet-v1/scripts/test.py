from pytorch_wheel_installer import find_links

links = find_links(
    distributions=("torch", "torchvision"),
    backend="cpu",
    language="py35",
    platform="linux",
)

print(links)