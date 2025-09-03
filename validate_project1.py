"""
We will import this script for testing your models.
You have to write the function load_my_models(). It 
should take no arguments and return your full fine-tuned 
model and lora model that you achieved the best results 
with.

We expect the models to accept inputs with dimension 
[batch_size, 3, 224, 224] and return outputs with 
dimension [1, num_classes]. Since you are using
ImageWoof, num_classes=10.

Run this script with 'python validate_project1.py' to 
check that your implementation of load_my_models() works 
as intended.
"""

import torch
# import your own modules


# Get your fine-tuned models here.
def load_my_models():
    """
    Loads the fine-tuned models (full and LoRA).
    These should be saved in the output folder.
    Return the models in the order indicated below,
    so the teachers can test them.
    """

    raise NotImplementedError()
    return full_model, lora_model


def test_load_my_models():
    full_model, lora_model = load_my_models()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_model = full_model.to(device)
    lora_model = lora_model.to(device)

    full_model.eval()
    lora_model.eval()

    # Send an example through the models, to check that they loaded properly
    test_img = torch.load('output/test_img.pth')
    with torch.no_grad():
        _ = full_model(test_img.unsqueeze(0).to(device))
        _ = lora_model(test_img.unsqueeze(0).to(device))


if __name__ == '__main__':
    test_load_my_models()