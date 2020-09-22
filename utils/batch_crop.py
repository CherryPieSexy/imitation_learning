import torch
import torch.nn.functional as fun


def batch_crop(tensor, pad_size=4):
    # tensor is assumed to have [t, b, c, h, w] shape
    time, batch, c, h, w = tensor.size()
    n = time * batch
    cropped = torch.zeros(
        n, c, h, w,
        dtype=torch.float32,
        device=tensor.device
    )
    # fun.pad(...) can only work with 4-d tensors => need to unsqueeze and squeeze back
    padded = tensor.view(n, c, h, w)
    padded = fun.pad(padded, [pad_size] * 4, mode='replicate')
    start_x = torch.randint(0, pad_size - 1, size=(n,))
    start_y = torch.randint(0, pad_size - 1, size=(n,))

    for i, (img, x, y) in enumerate(zip(padded, start_x, start_y)):
        cropped[i] = img[:, x:x + h, y:y + w]
    return cropped.view(time, batch, c, h, w)
